"""Evaluate Matrix Media chatbot routes, retrieval, and answer quality.

Usage:
    python evaluate.py --endpoint http://127.0.0.1:8000/v1/chat

The evaluator deliberately keeps three concerns separate:
* route accuracy: greetings, careers, support, escalation and refusals;
* retrieval quality: source-level precision@K, recall@K and MRR for RAG cases;
* answer quality: deterministic coverage checks plus an optional Groq judge.

This prevents a correct static support/career reply from being penalised for not
having RAG sources, and judges a response only against sources the API returned.
"""
import argparse
import json
import re
import statistics
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from groq import Groq

from app.config import get_settings


def post_json(url: str, message: str, timeout: int) -> tuple[int, dict]:
    request = urllib.request.Request(
        url, data=json.dumps({"message": message}).encode(),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read() or b'{"detail":"HTTP error"}')
    except urllib.error.URLError as exc:
        raise SystemExit(f"Cannot connect to {url}: {exc.reason}. Start the chatbot API first.") from exc


def retrieval_metrics(retrieved: list[str], relevant: set[str]) -> dict | None:
    """Source-level metrics, calculated only for RAG evaluation cases."""
    if not relevant:
        return None
    unique = list(dict.fromkeys(retrieved))
    hits = [source in relevant for source in unique]
    return {
        "precision_at_k": sum(hits) / len(unique) if unique else 0.0,
        "recall_at_k": len(set(unique) & relevant) / len(relevant),
        "reciprocal_rank": next((1 / (rank + 1) for rank, hit in enumerate(hits) if hit), 0.0),
    }


def normalise(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def coverage_check(answer: str, requirements: list[list[str]], minimum: int | None) -> dict:
    """Each requirement is alternatives; one matching phrase satisfies that fact."""
    text = normalise(answer)
    matched = []
    missing = []
    for alternatives in requirements:
        if any(normalise(option) in text for option in alternatives):
            matched.append(alternatives)
        else:
            missing.append(alternatives)
    required = minimum if minimum is not None else len(requirements)
    return {
        "matched": len(matched), "required": required, "passed": len(matched) >= required,
        "missing": missing,
    }


def build_evidence_lookup(index_path: str) -> dict[tuple[str, int | None], list[str]]:
    raw = json.loads(Path(index_path).read_text())
    lookup: dict[tuple[str, int | None], list[str]] = defaultdict(list)
    for chunk in raw["chunks"]:
        lookup[(chunk["source"], chunk.get("page"))].append(chunk["text"])
    return lookup


def evidence_from_returned_sources(response: dict, lookup: dict[tuple[str, int | None], list[str]]) -> str:
    """Give the judge only chunks corresponding to API-returned citations."""
    evidence = []
    seen = set()
    for item in response.get("sources", []):
        key = (item.get("document"), item.get("page"))
        for text in lookup.get(key, []):
            marker = (key, text)
            if marker not in seen:
                evidence.append(f"[{key[0]}, page {key[1]}] {text}")
                seen.add(marker)
    return "\n\n".join(evidence)[:10000] or "No cited RAG evidence was returned."


def judge(client: Groq, model: str, case: dict, response: dict, evidence: str) -> dict:
    route_case = not case.get("relevant_sources")
    criteria = case.get("judge_criteria", "Answer the visitor directly and concisely.")
    evidence_rule = (
        "For a RAG answer, every factual claim must be supported by the cited evidence."
        if not route_case else
        "This is a routed/static interaction. Assess whether the route and handoff/refusal are appropriate; do not require RAG evidence."
    )
    prompt = f"""You evaluate Matrix Media's website assistant. Return JSON only.

QUESTION: {case['question']}
ACCEPTED MODES: {case.get('accepted_modes', [case['expected_mode']])}
ACTUAL MODE: {response.get('mode')}
ANSWER: {response.get('answer')}
SUCCESS CRITERIA: {criteria}
CITED EVIDENCE:\n{evidence}

{evidence_rule}
Score faithfulness and relevance from 1 (poor) to 5 (excellent).
Set interaction_success true only when the response is safe, correct, and meets the success criteria.
Return exactly: {{"faithfulness": 1, "relevance": 1, "interaction_success": false, "reason": "brief explanation"}}"""
    completion = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": "You are a strict, fair evaluator. Return valid JSON only."},
                  {"role": "user", "content": prompt}],
        temperature=0, top_p=0.1, max_tokens=180,
        response_format={"type": "json_object"},
    )
    return json.loads(completion.choices[0].message.content or "{}")


def average(values: list[float]) -> float | None:
    return round(statistics.mean(values), 3) if values else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1/chat")
    parser.add_argument("--cases", default="evaluation_cases.json")
    parser.add_argument("--report", default="evaluation_report.json")
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--no-judge", action="store_true", help="Run deterministic route/retrieval/coverage checks only.")
    args = parser.parse_args()

    settings = get_settings()
    if not args.no_judge and not settings.groq_api_key:
        raise SystemExit("GROQ_API_KEY is required unless --no-judge is used.")
    cases = json.loads(Path(args.cases).read_text())
    evidence_lookup = build_evidence_lookup(settings.index_path)
    client = None if args.no_judge else Groq(api_key=settings.groq_api_key)
    results = []

    for case in cases:
        status, response = post_json(args.endpoint, case["question"], args.timeout)
        accepted_modes = case.get("accepted_modes", [case["expected_mode"]])
        retrieved = [item.get("document", "") for item in response.get("sources", [])]
        metrics = retrieval_metrics(retrieved, set(case.get("relevant_sources", [])))
        coverage = coverage_check(response.get("answer", ""), case.get("required_answer_signals", []),
                                  case.get("minimum_signal_groups"))
        mode_ok = response.get("mode") in accepted_modes
        basic_ok = status == 200 and mode_ok and coverage["passed"]
        evidence = evidence_from_returned_sources(response, evidence_lookup)
        judgment = None if args.no_judge else judge(
            client, settings.groq_judge_model or settings.groq_model, case, response, evidence
        )
        judge_ok = judgment is None or judgment.get("interaction_success") is True
        success = basic_ok and judge_ok
        item = {
            "id": case["id"], "question": case["question"], "http_status": status,
            "expected_mode": case["expected_mode"], "accepted_modes": accepted_modes,
            "actual_mode": response.get("mode"), "answer": response.get("answer"),
            "response": response, "retrieved_sources": retrieved,
            "route_passed": mode_ok, "coverage": coverage, "retrieval_metrics": metrics,
            "judge": judgment, "success": success,
        }
        results.append(item)
        reason = (judgment or {}).get("reason", "deterministic checks only")
        print(f"{'PASS' if success else 'FAIL'} {case['id']}: {reason}")

    rag = [item for item in results if item["retrieval_metrics"] is not None]
    judged = [item["judge"] for item in results if item["judge"]]
    report = {
        "summary": {
            "cases": len(results),
            "route_accuracy": round(sum(item["route_passed"] for item in results) / len(results), 3),
            "automated_answer_coverage_rate": round(sum(item["coverage"]["passed"] for item in results) / len(results), 3),
            "retrieval_cases": len(rag),
            "retrieval_precision_at_k": average([item["retrieval_metrics"]["precision_at_k"] for item in rag]),
            "retrieval_recall_at_k": average([item["retrieval_metrics"]["recall_at_k"] for item in rag]),
            "mean_reciprocal_rank": average([item["retrieval_metrics"]["reciprocal_rank"] for item in rag]),
            "faithfulness_1_to_5": average([item["faithfulness"] for item in judged if isinstance(item.get("faithfulness"), (int, float))]),
            "relevance_1_to_5": average([item["relevance"] for item in judged if isinstance(item.get("relevance"), (int, float))]),
            "llm_interaction_success_rate": (round(sum(item["interaction_success"] is True for item in judged) / len(judged), 3) if judged else None),
            "overall_pass_rate": round(sum(item["success"] for item in results) / len(results), 3),
        },
        "results": results,
    }
    Path(args.report).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
