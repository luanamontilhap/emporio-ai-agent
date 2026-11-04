from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emporio_agent.policy_rag import PolicyRAG


class FakeEncoder:
    """Tiny deterministic encoder so tests never download a model."""
    def encode(self, sentences, **kwargs):
        vectors = []
        for sentence in sentences:
            text = sentence.casefold()
            vectors.append([
                float("pix" in text or "pagamento" in text),
                float("devolução" in text or "arrependimento" in text),
                float("garantia" in text or "defeito" in text),
                float("cobre" in text),
            ])
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / np.maximum(norms, 1e-12)


def make_rag():
    return PolicyRAG(
        PROJECT_ROOT / "documents" / "politicas_da_loja.pdf",
        encoder=FakeEncoder(),
        chunk_size=800,
        chunk_overlap=100,
    )


def test_policy_pdf_is_extracted_and_chunked():
    rag = make_rag()
    assert len(rag.chunks) > 5
    assert all(chunk.page >= 1 for chunk in rag.chunks)
    full_text = " ".join(chunk.text.casefold() for chunk in rag.chunks)
    assert "arrependimento" in full_text
    assert "garantia" in full_text


def test_retrieval_returns_page_citations_and_relevant_text():
    result = make_rag().retrieve("Qual o prazo para devolução por arrependimento?", top_k=3)
    assert result["ok"]
    assert result["found_evidence"]
    assert all(item["page"] >= 1 for item in result["results"])
    assert any("devolução" in item["text"].casefold() for item in result["results"])


def test_retrieval_validates_empty_question_and_top_k():
    rag = make_rag()
    assert not rag.retrieve(" ")["ok"]
    assert not rag.retrieve("garantia", top_k=20)["ok"]


def test_context_builder_marks_source_and_page():
    rag = make_rag()
    result = rag.retrieve("garantia contra defeitos")
    context = rag.build_context(result)
    assert "politicas_da_loja.pdf" in context
    assert "page" in context


def test_context_builder_refuses_to_invent_when_no_evidence():
    context = PolicyRAG.build_context({"ok": True, "found_evidence": False, "results": []})
    assert "Do not infer or invent" in context


def test_chunking_keeps_policy_subsections_separate():
    rag = make_rag()
    sections = {chunk.section for chunk in rag.chunks}
    assert "4.1 Direito de Arrependimento (Compras Online)" in sections
    assert "5.1 Entregas na Região Metropolitana de Campo Grande" in sections
    assert "8.3 O que Não Cobre a Garantia" in sections
    # The delivery subsection must not be merged into the returns subsection.
    for chunk in rag.chunks:
        if chunk.section.startswith("4.1 "):
            assert "Frete grátis para pedidos" not in chunk.text


def test_lexical_ranking_finds_exact_policy_subsection():
    rag = make_rag()
    result = rag.retrieve("O que a garantia não cobre?", top_k=1, min_score=0)
    assert result["results"][0]["section"] == "8.3 O que Não Cobre a Garantia"
    assert result["results"][0]["lexical_score"] > 0
    assert "BM25" in result["retrieval"]["method"]
