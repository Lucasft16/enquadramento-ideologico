"""Testes para scoring/classifier e scoring/doc_graph."""

import pytest

from src.scoring.classifier import classify, _normalize, score_document_jaccard, score_document_graph
from src.scoring.doc_graph import build_doc_graph


def _make_model() -> dict:
    """Modelo mínimo controlado para testes."""
    return {
        "ideology_terms": {
            "neoliberal": {
                "mercado": 0.9, "privatização": 0.8, "eficiência": 0.7,
                "capital": 0.6, "investimento": 0.5,
            },
            "progressista": {
                "saúde": 0.9, "educação": 0.8, "trabalhador": 0.7,
                "direitos": 0.6, "social": 0.5,
            },
            "conservador": {
                "família": 0.9, "tradição": 0.8, "ordem": 0.7,
                "segurança": 0.6, "valores": 0.5,
            },
        },
        "graph_edges": [
            ("mercado", "privatização", 0.8),
            ("mercado", "eficiência", 0.7),
            ("privatização", "capital", 0.6),
            ("saúde", "educação", 0.9),
            ("saúde", "trabalhador", 0.7),
            ("família", "tradição", 0.8),
            ("família", "ordem", 0.6),
        ],
        "communities": [
            ["mercado", "privatização", "eficiência", "capital", "investimento"],
            ["saúde", "educação", "trabalhador", "direitos", "social"],
            ["família", "tradição", "ordem", "segurança", "valores"],
        ],
        "assignment": {"0": "neoliberal", "1": "progressista", "2": "conservador"},
        "vocab_size": 15,
    }


class TestNormalize:
    def test_sums_to_one(self):
        scores = {"a": 3.0, "b": 1.0, "c": 6.0}
        result = _normalize(scores)
        assert sum(result.values()) == pytest.approx(1.0)

    def test_all_zero_uniform(self):
        scores = {"a": 0.0, "b": 0.0}
        result = _normalize(scores)
        assert result["a"] == pytest.approx(0.5)
        assert result["b"] == pytest.approx(0.5)

    def test_single_ideology_is_one(self):
        result = _normalize({"x": 5.0})
        assert result["x"] == pytest.approx(1.0)


class TestClassifyJaccard:
    def test_distribution_sums_to_one(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência"]]
        scores = classify(windows, model, method="jaccard")
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_neoliberal_terms_dominant(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência", "capital", "investimento"]]
        scores = classify(windows, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "neoliberal"

    def test_progressista_terms_dominant(self):
        model = _make_model()
        windows = [["saúde", "educação", "trabalhador", "direitos", "social"]]
        scores = classify(windows, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "progressista"

    def test_conservador_terms_dominant(self):
        model = _make_model()
        windows = [["família", "tradição", "ordem", "segurança", "valores"]]
        scores = classify(windows, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "conservador"

    def test_empty_windows_uniform(self):
        model = _make_model()
        scores = classify([], model, method="jaccard")
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_unknown_terms_uniform(self):
        model = _make_model()
        scores = classify([["xyz", "abc", "naoexiste"]], model, method="jaccard")
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

class TestClassifyErrors:
    def test_invalid_method_raises(self):
        model = _make_model()
        with pytest.raises(ValueError):
            classify([["mercado"]], model, method="invalido")


class TestBuildDocGraph:
    def test_single_window_creates_edges(self):
        windows = [["mercado", "privatização", "eficiência"]]
        g = build_doc_graph(windows)
        assert g.has_edge("mercado", "privatização")
        assert g.has_edge("mercado", "eficiência")
        assert g.has_edge("privatização", "eficiência")

    def test_repeated_cooccurrence_accumulates_weight(self):
        windows = [["a", "b"], ["a", "b"], ["a", "b"]]
        g = build_doc_graph(windows)
        assert g.edge_weight("a", "b") == pytest.approx(3.0)

    def test_no_self_loops(self):
        windows = [["a", "a", "b"]]
        g = build_doc_graph(windows)
        assert not g.has_edge("a", "a")

    def test_empty_windows_returns_empty_graph(self):
        g = build_doc_graph([])
        assert g.num_vertices() == 0

    def test_single_token_window_no_edges(self):
        g = build_doc_graph([["mercado"]])
        assert g.num_edges() == 0


class TestScoreDocumentGraph:
    def test_distribution_sums_to_one(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência"]]
        scores = score_document_graph(windows, model)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_neoliberal_dominant(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência", "capital", "investimento"]]
        scores = score_document_graph(windows, model)
        assert max(scores, key=lambda k: scores[k]) == "neoliberal"

    def test_clustered_beats_scattered(self):
        """Termos ideológicos na mesma janela devem gerar score maior do que dispersos."""
        model = _make_model()
        clustered = [["mercado", "privatização", "capital"]]
        scattered = [["mercado"], ["privatização"], ["capital"]]
        s_clustered = score_document_graph(clustered, model)
        s_scattered = score_document_graph(scattered, model)
        assert s_clustered["neoliberal"] > s_scattered["neoliberal"]

    def test_empty_windows_uniform(self):
        model = _make_model()
        scores = score_document_graph([], model)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)


class TestClassifyGraph:
    def test_graph_method_default(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência"]]
        scores = classify(windows, model)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_jaccard_method_still_works(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência"]]
        scores = classify(windows, model, method="jaccard")
        assert max(scores, key=lambda k: scores[k]) == "neoliberal"
