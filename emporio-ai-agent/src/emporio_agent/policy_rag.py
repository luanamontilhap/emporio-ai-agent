from __future__ import annotations

"""Hybrid retrieval layer for the store policy manual.

The module extracts section-aware passages and combines multilingual dense
similarity with a lightweight lexical BM25 score. Answer generation remains
separate so evidence selection can be tested independently.
"""

from collections import Counter
from dataclasses import dataclass
from math import log
from pathlib import Path
import re
import unicodedata
from typing import Protocol

import numpy as np

DEFAULT_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
HEADING_RE = re.compile(r"^\s*(\d+(?:\.\d+)*\.?\s+[A-ZÁÉÍÓÚÂÊÔÃÕÇ].+?)\s*$")
PAGE_RE = re.compile(r"^(?:Página\s+\d+|Empório da Música Manual de Políticas e Procedimentos)$", re.I)
TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)
STOPWORDS = {
    "a", "as", "o", "os", "um", "uma", "de", "do", "da", "dos", "das",
    "em", "no", "na", "nos", "nas", "por", "para", "com", "e", "ou",
    "que", "qual", "quais", "como", "quando", "onde", "se", "ao", "aos",
    "me", "minha", "meu", "loja", "cliente", "sobre", "tem", "ter",
}


class Encoder(Protocol):
    def encode(self, sentences, **kwargs): ...


@dataclass(frozen=True)
class PolicyChunk:
    chunk_id: int
    text: str
    page: int
    section: str


def _normalize_token_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text.casefold())
    return "".join(char for char in text if not unicodedata.combining(char))


def _tokens(text: str) -> list[str]:
    return [token for token in TOKEN_RE.findall(_normalize_token_text(text))
            if token not in STOPWORDS and len(token) > 1]


class PolicyRAG:
    """Load, section-chunk, embed and hybrid-rank policy evidence."""

    def __init__(self, pdf_path: str | Path, model_name: str = DEFAULT_MODEL,
                 encoder: Encoder | None = None, chunk_size: int = 1100,
                 chunk_overlap: int = 160, dense_weight: float = 0.70):
        self.pdf_path = Path(pdf_path)
        if not self.pdf_path.exists():
            raise FileNotFoundError(f"Policy PDF not found: {self.pdf_path}")
        if chunk_size < 200:
            raise ValueError("chunk_size must be at least 200 characters")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_size")
        if not 0.0 <= dense_weight <= 1.0:
            raise ValueError("dense_weight must be between 0 and 1")

        self.dense_weight = dense_weight
        self.chunks = self._load_and_chunk(chunk_size, chunk_overlap)
        if not self.chunks:
            raise ValueError(f"No readable text was extracted from {self.pdf_path}")
        self.encoder = encoder or self._load_encoder(model_name)
        self.model_name = model_name
        self.embeddings = self._encode([f"{chunk.section}\n{chunk.text}" for chunk in self.chunks])
        self._tokenized_chunks = [_tokens(chunk.section + " " + chunk.text) for chunk in self.chunks]
        self._avg_doc_length = max(1.0, sum(map(len, self._tokenized_chunks)) / len(self._tokenized_chunks))
        self._document_frequency = Counter()
        for tokens in self._tokenized_chunks:
            self._document_frequency.update(set(tokens))

    @staticmethod
    def _load_encoder(model_name: str) -> Encoder:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise RuntimeError("RAG dependencies are missing. Install requirements.txt first.") from exc
        return SentenceTransformer(model_name)

    def _load_and_chunk(self, chunk_size: int, chunk_overlap: int) -> list[PolicyChunk]:
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError("PyMuPDF is missing. Install requirements.txt first.") from exc

        document = fitz.open(self.pdf_path)
        sections: list[dict] = []
        current_section = "General"
        for page_index, page in enumerate(document):
            pending_bullet = False
            for raw_line in page.get_text("text").splitlines():
                line = re.sub(r"\s+", " ", raw_line).strip()
                if not line or PAGE_RE.match(line):
                    continue
                if line in {"•", "•​", "•\u200b"}:
                    pending_bullet = True
                    continue
                heading = HEADING_RE.match(line)
                if heading and len(line) < 150:
                    current_section = heading.group(1)
                    sections.append({"section": current_section, "page": None, "parts": []})
                    pending_bullet = False
                    continue
                if not sections or sections[-1]["section"] != current_section:
                    sections.append({"section": current_section, "page": None, "parts": []})
                parts = sections[-1]["parts"]
                if not parts:
                    sections[-1]["page"] = page_index + 1
                if pending_bullet:
                    parts.append("• " + line)
                    pending_bullet = False
                elif parts and not parts[-1].endswith((".", ":", ";")):
                    parts[-1] += " " + line
                else:
                    parts.append(line)
        document.close()

        chunks: list[PolicyChunk] = []
        for section_data in sections:
            text = "\n".join(section_data["parts"]).strip()
            if not text:
                continue
            start = 0
            while start < len(text):
                end = min(start + chunk_size, len(text))
                if end < len(text):
                    boundary = max(text.rfind("\n", start, end), text.rfind(" ", start, end))
                    if boundary > start + int(chunk_size * 0.60):
                        end = boundary
                piece = text[start:end].strip()
                if piece:
                    chunks.append(PolicyChunk(len(chunks), piece, section_data["page"], section_data["section"]))
                if end >= len(text):
                    break
                start = max(end - chunk_overlap, start + 1)
        return chunks

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.encoder.encode(texts, convert_to_numpy=True,
                                      normalize_embeddings=True, show_progress_bar=False)
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or matrix.shape[0] != len(texts):
            raise ValueError("The embedding model returned an invalid matrix shape")
        return matrix

    def _lexical_scores(self, query_tokens: list[str]) -> np.ndarray:
        """BM25 scores normalized to [0, 1]."""
        scores = np.zeros(len(self.chunks), dtype=np.float32)
        if not query_tokens:
            return scores
        n_docs, k1, b = len(self.chunks), 1.5, 0.75
        query_frequency = Counter(query_tokens)
        for index, tokens in enumerate(self._tokenized_chunks):
            frequencies = Counter(tokens)
            length = len(tokens)
            for term, qf in query_frequency.items():
                tf = frequencies.get(term, 0)
                if not tf:
                    continue
                df = self._document_frequency.get(term, 0)
                idf = log(1 + (n_docs - df + 0.5) / (df + 0.5))
                denominator = tf + k1 * (1 - b + b * length / self._avg_doc_length)
                scores[index] += idf * (tf * (k1 + 1) / denominator) * (1 + log(qf))
        maximum = float(scores.max()) if scores.size else 0.0
        return scores / maximum if maximum > 0 else scores

    def retrieve(self, question: str, top_k: int = 4, min_score: float = 0.20) -> dict:
        """Return hybrid-ranked passages with source page and section metadata."""
        question = question.strip()
        if not question:
            return {"ok": False, "error": "Question cannot be empty.", "results": []}
        if top_k < 1 or top_k > 10:
            return {"ok": False, "error": "top_k must be between 1 and 10.", "results": []}
        if not 0.0 <= min_score <= 1.0:
            return {"ok": False, "error": "min_score must be between 0 and 1.", "results": []}

        query_vector = self._encode([question])[0]
        dense_scores = np.clip(self.embeddings @ query_vector, 0.0, 1.0)
        lexical_scores = self._lexical_scores(_tokens(question))
        combined = self.dense_weight * dense_scores + (1.0 - self.dense_weight) * lexical_scores
        ranked_indices = np.argsort(combined)[::-1]
        results = []
        for index in ranked_indices:
            score = float(combined[index])
            if score < min_score:
                break
            chunk = self.chunks[int(index)]
            results.append({"chunk_id": chunk.chunk_id, "text": chunk.text,
                            "page": chunk.page, "section": chunk.section,
                            "score": round(score, 4),
                            "dense_score": round(float(dense_scores[index]), 4),
                            "lexical_score": round(float(lexical_scores[index]), 4),
                            "source": self.pdf_path.name})
            if len(results) >= top_k:
                break
        return {"ok": True, "question": question, "found_evidence": bool(results),
                "results": results,
                "retrieval": {"method": "hybrid multilingual embeddings + BM25 lexical ranking",
                              "model": self.model_name, "dense_weight": self.dense_weight,
                              "lexical_weight": round(1.0 - self.dense_weight, 2),
                              "top_k": top_k, "min_score": min_score}}

    @staticmethod
    def build_context(retrieval_result: dict) -> str:
        """Format retrieved passages for a later grounded LLM prompt."""
        if not retrieval_result.get("ok") or not retrieval_result.get("found_evidence"):
            return "No relevant policy evidence was retrieved. Do not infer or invent a policy."
        return "\n\n---\n\n".join(
            f"[Source: {item['source']}, page {item['page']}, section: {item['section']}, relevance: {item['score']}]\n{item['text']}"
            for item in retrieval_result["results"]
        )
