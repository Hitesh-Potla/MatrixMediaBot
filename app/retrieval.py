import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import chromadb
from fastembed import TextEmbedding
from fastembed.rerank.cross_encoder import TextCrossEncoder


@dataclass
class RetrievedChunk:
    text: str
    source: str
    page: int | None
    score: float
    hybrid_score: float
    rerank_score: float


TOKEN_RE = re.compile(r"[a-z0-9]{2,}")
STOP_WORDS = {
    "a", "an", "and", "are", "at", "about", "can", "do", "does", "for", "from", "how",
    "i", "in", "is", "it", "me", "of", "on", "our", "please", "tell", "the", "to", "us",
    "what", "when", "where", "which", "who", "will", "with", "you", "your", "matrix", "media",
}


def tokens(text: str) -> list[str]:
    normalized = []
    for word in TOKEN_RE.findall(text.lower()):
        if word.endswith("ies") and len(word) > 4:
            word = f"{word[:-3]}y"
        elif word.endswith("s") and not word.endswith("ss") and len(word) > 3:
            word = word[:-1]
        if word not in STOP_WORDS:
            normalized.append(word)
    return normalized


class Retriever:
    def __init__(self, index_path: str, embedding_model: str, cache_dir: str, chroma_path: str,
                 collection_name: str, reranker_model: str, reranker_candidate_count: int,
                 local_files_only: bool = True):
        raw = json.loads(Path(index_path).read_text())
        self.chunks = raw["chunks"]
        self.index_by_id = {chunk["id"]: index for index, chunk in enumerate(self.chunks)}
        self.chroma = chromadb.PersistentClient(path=chroma_path)
        self.collection = self.chroma.get_collection(collection_name)
        if self.collection.count() != len(self.chunks):
            raise RuntimeError("Chroma collection and chunk metadata index are out of sync. Run scripts/ingest.py.")
        # A title is deliberately repeated in the lexical representation so document
        # names such as "Audience Segment" can resolve direct visitor terminology.
        self.lexical_docs = [tokens(
            f"{chunk.get('title', chunk['source'])} " * 4 +
            f"{chunk.get('section', '')} " * 2 + chunk["text"]
        )
                             for chunk in self.chunks]
        self.doc_lengths = np.asarray([len(doc) for doc in self.lexical_docs], dtype=np.float32)
        self.average_doc_length = float(self.doc_lengths.mean()) or 1.0
        self.document_frequency = Counter(token for doc in self.lexical_docs for token in set(doc))
        self.embedder = TextEmbedding(
            model_name=embedding_model, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.reranker = TextCrossEncoder(
            model_name=reranker_model, cache_dir=cache_dir, local_files_only=local_files_only
        )
        self.reranker_candidate_count = max(5, reranker_candidate_count)

    def _bm25_scores(self, question: str) -> np.ndarray:
        query_tokens = tokens(question)
        scores = np.zeros(len(self.lexical_docs), dtype=np.float32)
        if not query_tokens:
            return scores
        total = len(self.lexical_docs)
        k1, b = 1.5, 0.75
        for index, document in enumerate(self.lexical_docs):
            frequencies = Counter(document)
            for token in query_tokens:
                df = self.document_frequency.get(token, 0)
                if not df:
                    continue
                tf = frequencies[token]
                if not tf:
                    continue
                idf = math.log(1 + (total - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (1 - b + b * self.doc_lengths[index] / self.average_doc_length)
                scores[index] += idf * (tf * (k1 + 1) / denominator)

            # Documents such as "Audience Segment" span many pages. If a visitor
            # explicitly names that topic, page openings are more likely to contain
            # the audience/category heading than a mid-page service detail.
            title_overlap = len(set(query_tokens) & set(tokens(self.chunks[index].get("title", ""))))
            if title_overlap and self.chunks[index].get("page_start") and self.chunks[index].get("section"):
                scores[index] += 2.5 * title_overlap
        return scores

    def search(self, question: str, limit: int = 5) -> list[RetrievedChunk]:
        query = np.fromiter(next(self.embedder.embed([question])), dtype=np.float32)
        query /= max(np.linalg.norm(query), 1e-12)
        # Chroma owns persisted embeddings and performs the dense nearest-neighbour query.
        dense_result = self.collection.query(
            query_embeddings=[query.tolist()], n_results=len(self.chunks), include=["distances"]
        )
        dense_scores = np.full(len(self.chunks), -1.0, dtype=np.float32)
        for chunk_id, distance in zip(dense_result["ids"][0], dense_result["distances"][0]):
            dense_scores[self.index_by_id[chunk_id]] = 1.0 - float(distance)
        lexical_scores = self._bm25_scores(question)

        # Reciprocal-rank fusion avoids fragile score-scale comparisons between
        # embeddings and BM25 while retaining dense search for paraphrased questions.
        dense_order = np.argsort(dense_scores)[::-1]
        dense_rank = np.empty(len(dense_scores), dtype=np.int32)
        dense_rank[dense_order] = np.arange(len(dense_scores))
        fused_scores = 0.70 / (60 + dense_rank)
        if lexical_scores.max() > 0:
            lexical_order = np.argsort(lexical_scores)[::-1]
            lexical_rank = np.empty(len(lexical_scores), dtype=np.int32)
            lexical_rank[lexical_order] = np.arange(len(lexical_scores))
            fused_scores += 0.30 / (60 + lexical_rank)

        # Semantic overlap deliberately repeats boundary text. Do not spend the
        # prompt budget on nearly identical chunks from the same passage.
        selected: list[int] = []
        query_terms = set(tokens(question))
        # A long document can define several labelled sub-audiences. When the
        # question explicitly matches the document's title, include its labelled
        # section openings first; these contain the category names a visitor seeks.
        section_anchors = [
            index for index, chunk in enumerate(self.chunks)
            if chunk.get("page_start") and chunk.get("section")
            and query_terms.intersection(tokens(chunk.get("title", "")))
        ]
        # Keep up to three labelled anchors (for example Startups, SMEs, and
        # Enterprise) even when the final top-K is five.
        for index in sorted(section_anchors, key=lambda item: fused_scores[item], reverse=True)[:min(3, limit - 2)]:
            selected.append(int(index))

        # Rerank only the leading hybrid candidates using a cross-encoder. Unlike
        # embedding similarity, this model reads the question and chunk together.
        candidate_indices = list(dict.fromkeys(
            selected + [int(index) for index in np.argsort(fused_scores)[::-1][:self.reranker_candidate_count]]
        ))
        rerank_inputs = [
            f"Document: {self.chunks[index].get('title', '')}. "
            f"Section: {self.chunks[index].get('section', '')}.\n{self.chunks[index]['text']}"
            for index in candidate_indices
        ]
        rerank_scores = list(self.reranker.rerank(question, rerank_inputs))
        rerank_by_index = dict(zip(candidate_indices, map(float, rerank_scores)))
        reranked_order = sorted(candidate_indices, key=lambda index: rerank_by_index[index], reverse=True)

        # Fetch only candidate vectors from Chroma for duplicate suppression; the
        # app never keeps a full embedding matrix in memory.
        candidate_vectors = self.collection.get(
            ids=[self.chunks[index]["id"] for index in candidate_indices], include=["embeddings"]
        )
        vectors_by_id = {
            chunk_id: np.asarray(vector, dtype=np.float32)
            for chunk_id, vector in zip(candidate_vectors["ids"], candidate_vectors["embeddings"])
        }
        # Retain special section anchors first, then use cross-encoder ordering for
        # all remaining candidates. This keeps broad section-navigation answers intact.
        selected = sorted(selected, key=lambda index: rerank_by_index[index], reverse=True)
        for index in reranked_order:
            if int(index) in selected:
                continue
            candidate_vector = vectors_by_id.get(self.chunks[index]["id"])
            if candidate_vector is not None and any(
                float(candidate_vector @ vectors_by_id[self.chunks[other]["id"]]) > 0.94
                for other in selected if self.chunks[other]["id"] in vectors_by_id
            ):
                continue
            selected.append(int(index))
            if len(selected) == limit:
                break
        return [
            RetrievedChunk(
                text=self.chunks[i]["text"], source=self.chunks[i]["source"],
                page=self.chunks[i].get("page"), score=float(dense_scores[i]),
                hybrid_score=float(fused_scores[i]),
                rerank_score=rerank_by_index[i],
            )
            for i in selected
        ]
