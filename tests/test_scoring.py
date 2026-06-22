"""Testes para scoring/classifier e scoring/doc_graph."""

import pytest

from src.scoring.classifier import classify, _normalize, score_document_jaccard, score_document_graph
from src.scoring.doc_graph import build_doc_graph


def _make_model() -> dict:
    """Modelo mínimo controlado para testes."""
    return {
        "ideology_terms": {
            "libertarianismo": {
                "mercado": 0.9, "privatização": 0.8, "eficiência": 0.7,
                "capital": 0.6, "investimento": 0.5,
            },
            "social-democracia": {
                "saúde": 0.9, "educação": 0.8, "trabalhador": 0.7,
                "direitos": 0.6, "social": 0.5,
            },
            "conservadorismo": {
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
        "assignment": {"0": "libertarianismo", "1": "social-democracia", "2": "conservadorismo"},
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
        
    def test_normalize_empty_dict(self):
        """_normalize de dicionário vazio deve retornar {}."""
        assert _normalize({}) == {}
    
    def test_normalize_all_negative(self):
        """Scores todos negativos: total <= 0 → distribuição uniforme."""
        result = _normalize({"A": -1.0, "B": -2.0})
        assert result["A"] == pytest.approx(0.5)
        assert result["B"] == pytest.approx(0.5)
    
    def test_normalize_large_values(self):
        """Valores muito grandes: normalização deve funcionar sem overflow."""
        import sys
        result = _normalize({"A": sys.float_info.max / 2, "B": sys.float_info.max / 2})
        assert result["A"] == pytest.approx(0.5, abs=1e-6)


class TestClassifyJaccard:
    def test_distribution_sums_to_one(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência"]]
        scores = classify(windows, model, method="jaccard")
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_libertarianismo_terms_dominant(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência", "capital", "investimento"]]
        scores = classify(windows, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "libertarianismo"

    def test_social_democracia_terms_dominant(self):
        model = _make_model()
        windows = [["saúde", "educação", "trabalhador", "direitos", "social"]]
        scores = classify(windows, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "social-democracia"

    def test_conservadorismo_terms_dominant(self):
        model = _make_model()
        windows = [["família", "tradição", "ordem", "segurança", "valores"]]
        scores = classify(windows, model, method="jaccard")
        dominant = max(scores, key=lambda k: scores[k])
        assert dominant == "conservadorismo"

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
        
    def test_empty_model_ideologies(self):
        """Modelo sem ideologias → _normalize retorna {} sem divisão por zero."""
        result = _normalize({})
        assert result == {}
    
    def test_single_ideology_model(self):
        """Modelo com uma única ideologia → probabilidade = 1.0."""
        model = {
            "ideology_terms": {"unica": {"mercado": 1.0, "capital": 0.8}},
            "graph_edges": [("mercado", "capital", 0.9)],
        }
        scores = classify([["mercado", "capital"]], model, method="jaccard")
        assert scores["unica"] == pytest.approx(1.0)
    
    def test_terms_overlap_all_ideologies_equally(self):
        """Termos que aparecem igualmente em todas as ideologias → distribuição uniforme."""
        # Mesmo conjunto de termos em todas as ideologias com mesmos scores
        terms = {"x": 1.0, "y": 1.0}
        model = {
            "ideology_terms": {"A": dict(terms), "B": dict(terms), "C": dict(terms)},
            "graph_edges": [],
        }
        scores = classify([["x", "y"]], model, method="jaccard")
        assert scores["A"] == pytest.approx(scores["B"], abs=1e-9)
        assert scores["B"] == pytest.approx(scores["C"], abs=1e-9)
    
    def test_jaccard_model_with_no_graph_edges(self):
        """Jaccard funciona sem graph_edges no modelo."""
        model = {
            "ideology_terms": {
                "esquerda": {"trabalho": 0.9, "sindicato": 0.8},
                "direita": {"mercado": 0.9, "privatizar": 0.8},
            },
            "graph_edges": [],
        }
        scores = classify([["trabalho", "sindicato"]], model, method="jaccard")
        assert scores["esquerda"] > scores["direita"]
    
    def test_very_long_terms_list(self):
        """Lista de 10.000 termos (todos no modelo) não deve ser lenta ou falhar."""
        ideology_terms = {f"t{i}": float(i) for i in range(10000)}
        model = {
            "ideology_terms": {"grande": ideology_terms},
            "graph_edges": [],
        }
        terms = [f"t{i}" for i in range(10000)]
        scores = classify([terms], model, method="jaccard")
        assert scores["grande"] == pytest.approx(1.0)
    
    def test_duplicate_terms_in_list(self):
        """Termos duplicados na lista de entrada: Jaccard usa set() internamente."""
        model = {
            "ideology_terms": {
                "A": {"mercado": 1.0},
                "B": {"saúde": 1.0},
            },
            "graph_edges": [],
        }
        # "mercado" repetido 100x: deve se comportar igual a ["mercado"]
        terms_repeated = ["mercado"] * 100
        terms_single = ["mercado"]
        s1 = classify([terms_repeated], model, method="jaccard")
        s2 = classify([terms_single], model, method="jaccard")
        assert s1["A"] == pytest.approx(s2["A"])


class TestScoreDocumentGraph:
    def test_distribution_sums_to_one(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência"]]
        scores = score_document_graph(windows, model)
        assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9)

    def test_libertarianismo_dominant(self):
        model = _make_model()
        windows = [["mercado", "privatização", "eficiência", "capital", "investimento"]]
        scores = score_document_graph(windows, model)
        assert max(scores, key=lambda k: scores[k]) == "libertarianismo"

    def test_clustered_beats_scattered(self):
        """Termos ideológicos na mesma janela devem gerar score maior do que dispersos."""
        model = _make_model()
        clustered = [["mercado", "privatização", "capital"]]
        scattered = [["mercado"], ["privatização"], ["capital"]]
        s_clustered = score_document_graph(clustered, model)
        s_scattered = score_document_graph(scattered, model)
        assert s_clustered["libertarianismo"] > s_scattered["libertarianismo"]

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
        assert max(scores, key=lambda k: scores[k]) == "libertarianismo"
