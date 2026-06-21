"""Testes para scoring/classifier: distribuição soma 1 e aponta ideologia esperada."""

import pytest

from src.scoring.classifier import classify, _normalize, score_document_jaccard


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
        terms = ["mercado", "privatização", "eficiência"]
        scores = classify(terms, model, method="jaccard")
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_neoliberal_terms_dominant(self):
        model = _make_model()
        terms = ["mercado", "privatização", "eficiência", "capital", "investimento"]
        scores = classify(terms, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "neoliberal"

    def test_progressista_terms_dominant(self):
        model = _make_model()
        terms = ["saúde", "educação", "trabalhador", "direitos", "social"]
        scores = classify(terms, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "progressista"

    def test_conservador_terms_dominant(self):
        model = _make_model()
        terms = ["família", "tradição", "ordem", "segurança", "valores"]
        scores = classify(terms, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "conservador"

    def test_empty_terms_uniform(self):
        model = _make_model()
        scores = classify([], model, method="jaccard")
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_unknown_terms_uniform(self):
        model = _make_model()
        scores = classify(["xyz", "abc", "naoexiste"], model, method="jaccard")
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

class TestClassifyErrors:
    def test_invalid_method_raises(self):
        model = _make_model()
        with pytest.raises(ValueError):
            classify(["mercado"], model, method="invalido")
