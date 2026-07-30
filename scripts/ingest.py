"""One-time, local embedding build. Run from the repository root."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
import chromadb
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder
from pypdf import PdfReader
from tokenizers import Tokenizer

EXCLUDED_SOURCE_FILES = {"requirements.txt"}


def count_tokens(tokenizer: Tokenizer, text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False).ids)


def split_token_window(text: str, tokenizer: Tokenizer, max_tokens: int) -> list[str]:
    """Last-resort split for an unusually long sentence; normal chunks keep sentences whole."""
    ids = tokenizer.encode(text, add_special_tokens=False).ids
    return [tokenizer.decode(ids[start:start + max_tokens], skip_special_tokens=True).strip()
            for start in range(0, len(ids), max_tokens) if ids[start:start + max_tokens]]


def atomic_units(text: str, tokenizer: Tokenizer, max_tokens: int = 90) -> list[str]:
    """Create paragraph/sentence units without exceeding the embedding-token budget."""
    text = re.sub(r"[ \t]+", " ", text).strip()
    paragraphs = [re.sub(r"\s*\n\s*", " ", p).strip() for p in re.split(r"\n\s*\n+", text)]
    paragraphs = [p for p in paragraphs if p]
    # PDF extraction often has no blank paragraphs. In that case sentences are safer units.
    if len(paragraphs) <= 1:
        paragraphs = re.split(r"(?<=[.!?])\s+", text)

    units = []
    for paragraph in paragraphs:
        sentences = re.split(r"(?<=[.!?])\s+", paragraph)
        current = ""
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            sentence_tokens = count_tokens(tokenizer, sentence)
            if sentence_tokens > max_tokens:
                if current:
                    units.append(current)
                    current = ""
                units.extend(split_token_window(sentence, tokenizer, max_tokens))
                continue
            candidate = f"{current} {sentence}".strip()
            if current and count_tokens(tokenizer, candidate) > max_tokens:
                units.append(current)
                current = sentence
            else:
                current = candidate
        if current:
            units.append(current)
    # return [unit for unit in units if len(unit) >= 40]
    merged = []
    pending_heading = None

    for unit in units:
        unit = unit.strip()

        if len(unit) < 40:
            pending_heading = unit
            continue

        if pending_heading:
            unit = f"{pending_heading}\n{unit}"
            pending_heading = None

        merged.append(unit)

    return merged


def semantic_chunks(text: str, model: TextEmbedding, tokenizer: Tokenizer, target_tokens: int = 170,
                    max_tokens: int = 220, overlap_units: int = 1):
    """Group adjacent sentence/paragraph units using local embedding similarity.

    A boundary is considered when neighbouring units are semantically less similar
    than the lowest quartile for that page. Chunk size remains bounded so context
    stays cheap and precise. Token limits use the exact BGE tokenizer, not a
    character approximation. This runs only during indexing, never per request.
    """
    units = atomic_units(text, tokenizer, max_tokens=max(48, max_tokens // 2))
    if not units:
        return
    if len(units) == 1:
        yield units[0]
        return

    vectors = np.asarray(list(model.embed(units)), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    adjacent_similarity = np.sum(vectors[:-1] * vectors[1:], axis=1)
    boundary_threshold = max(0.35, float(np.quantile(adjacent_similarity, 0.25)))

    current: list[str] = []
    current_tokens = 0
    for index, unit in enumerate(units):
        unit_tokens = count_tokens(tokenizer, unit)
        semantic_break = index > 0 and adjacent_similarity[index - 1] < boundary_threshold
        should_flush = current and (
            current_tokens + unit_tokens > max_tokens or
            (semantic_break and current_tokens >= target_tokens)
        )
        if should_flush:
            yield " ".join(current)
            current = current[-overlap_units:] if overlap_units else []
            current_tokens = sum(count_tokens(tokenizer, item) for item in current)
            # Preserve overlap only when it does not violate the hard token cap.
            if current_tokens + unit_tokens > max_tokens:
                current = []
                current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        yield " ".join(current)


def read_source(path: Path):
    if path.suffix.lower() == ".pdf":
        for number, page in enumerate(PdfReader(path).pages, start=1):
            yield page.extract_text() or "", number
    elif path.suffix.lower() == ".txt":
        yield path.read_text(encoding="utf-8", errors="replace"), None


def document_title(path: Path) -> str:
    """Turn a file name into useful retrieval metadata without exposing a file path."""
    title = path.stem.replace("Design & Development Notes - ", "")
    return re.sub(r"\s+", " ", title).strip()


def page_section(text: str) -> str:
    """Extract a labelled page section when a source document provides one."""
    compact = re.sub(r"\s+", " ", text).strip()
    match = re.search(r"\bPAGE\s+\d+\s*:\s*(.+?)(?=\s+(?:Hero Section|Suggested Heading)\b|$)", compact,
                      flags=re.IGNORECASE)
    return match.group(1).strip()[:120] if match else ""


def chroma_metadata(chunk: dict) -> dict:
    """Chroma metadata accepts scalar values only; use -1 for a TXT/no-page record."""
    return {
        "source": chunk["source"],
        "title": chunk["title"],
        "section": chunk["section"] or "unlabelled",
        "page": chunk["page"] if chunk["page"] is not None else -1,
        "page_start": chunk["page_start"],
        "content_token_count": chunk["content_token_count"],
    }


def write_chroma(records: list[dict], vectors: np.ndarray, path: str, collection_name: str) -> None:
    client = chromadb.PersistentClient(path=path)
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=collection_name, configuration={"hnsw": {"space": "cosine"}}
    )
    for start in range(0, len(records), 100):
        batch = records[start:start + 100]
        collection.add(
            ids=[record["id"] for record in batch],
            documents=[record["text"] for record in batch],
            embeddings=vectors[start:start + len(batch)].tolist(),
            metadatas=[chroma_metadata(record) for record in batch],
        )
    if collection.count() != len(records):
        raise RuntimeError("Chroma collection count does not match the generated chunk count.")


def main():
    parser = argparse.ArgumentParser()
    # The supplied PDFs/TXT files live in the project root by default.
    parser.add_argument("--source", default=".")
    parser.add_argument("--output", default="storage/index.json")
    parser.add_argument("--chroma-path", default="storage/chroma")
    parser.add_argument("--chroma-collection", default="matrix_media")
    parser.add_argument("--reranker-model", default="Xenova/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--cache-dir", default="storage/fastembed",
                        help="Persistent directory for the downloaded embedding model.")
    parser.add_argument("--recursive", action="store_true",
                        help="Also scan nested folders under --source.")
    parser.add_argument("--download-model", action="store_true",
                        help="Allow downloading the embedding model when it is not already cached.")
    parser.add_argument("--target-chunk-tokens", type=int, default=170,
                        help="Preferred BGE-token size before a semantic boundary closes a chunk.")
    parser.add_argument("--max-chunk-tokens", type=int, default=220,
                        help="Hard BGE-token limit for each chunk's source text.")
    args = parser.parse_args()
    if not 32 <= args.target_chunk_tokens <= args.max_chunk_tokens:
        raise SystemExit("target-chunk-tokens must be at least 32 and no greater than max-chunk-tokens.")
    model = TextEmbedding(model_name="BAAI/bge-small-en-v1.5", cache_dir=args.cache_dir,
                          local_files_only=not args.download_model)
    # Download/cache the local cross-encoder during setup/build. It is used only
    # at query time to rerank the small hybrid candidate set.
    TextCrossEncoder(model_name=args.reranker_model, cache_dir=args.cache_dir,
                     local_files_only=not args.download_model)
    tokenizer = model.model.tokenizer
    records = []
    source = Path(args.source)
    paths = source.rglob("*") if args.recursive else source.iterdir()
    for path in sorted(paths):
        if path.name in EXCLUDED_SOURCE_FILES or path.suffix.lower() not in {".pdf", ".txt"}:
            continue
        for text, page in read_source(path):
            section = page_section(text)
            parts = list(semantic_chunks(
                text, model, tokenizer, args.target_chunk_tokens, args.max_chunk_tokens
            ))
            for part_index, part in enumerate(parts):
                title = document_title(path)
                content_tokens = count_tokens(tokenizer, part)
                embedding_text = f"Document topic: {title}. Section: {section}.\n\n{part}"
                records.append({
                    "id": f"{path.stem}:{page or 0}:{part_index}",
                    "text": part,
                    "source": path.name,
                    "title": title,
                    "section": section,
                    "page": page,
                    "page_start": part_index == 0,
                    "chunk_index_on_page": part_index,
                    "chunks_on_page": len(parts),
                    "content_token_count": content_tokens,
                    "metadata": {
                        "document_title": title,
                        "section": section,
                        "page": page,
                        "chunk_index_on_page": part_index,
                        "content_token_count": content_tokens,
                    },
                    # The title anchors generic page text to its subject (for example,
                    # "Audience Segment"), but only the human-readable text is sent to Groq.
                    "embedding_text": embedding_text,
                    "embedding_token_count": count_tokens(tokenizer, embedding_text),
                })
    if not records:
        raise SystemExit("No readable .pdf or .txt content found.")
    vectors = np.asarray(list(model.embed([r["embedding_text"] for r in records])), dtype=np.float32)
    vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)
    write_chroma(records, vectors, args.chroma_path, args.chroma_collection)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "index_version": 3,
        "embedding_model": "BAAI/bge-small-en-v1.5",
        "chunking": {
            "strategy": "semantic_sentence_boundary",
            "target_content_tokens": args.target_chunk_tokens,
            "max_content_tokens": args.max_chunk_tokens,
            "overlap_units": 1,
        },
        "vector_store": {
            "type": "chroma",
            "path": args.chroma_path,
            "collection": args.chroma_collection,
            "distance": "cosine",
        },
        "chunks": records,
    }))
    print(f"Indexed {len(records)} chunks into Chroma (max {args.max_chunk_tokens} content tokens)")


if __name__ == "__main__":
    main()
