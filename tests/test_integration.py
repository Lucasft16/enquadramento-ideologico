"""Testes end-to-end: texto bruto → janelas → modelo → classificação."""

import pytest
from src.parser.pipeline import process_document, process_corpus
from src.graph_build.cooccurrence import count_cooccurrences, build_vocab_from_windows
from src.graph_build.weighting import build_weighted_graph
from src.scoring.classifier import classify


def _build_minimal_model(docs: list[str], seeds: dict) -> dict:
    """Pipeline mínimo sem spaCy para testes de integração."""
    from src.graph_build.weighting import build_weighted_graph
    from src.analysis.filtering import threshold_filter
    from src.analysis.centrality import degree_centrality
    from src.model.anchoring import anchor_communities, build_ideology_term_map
    from src.analysis.traversal import connected_components

    docs_windows = process_corpus(docs, use_lemmatizer=False, window_size=3)
    vocab = build_vocab_from_windows(docs_windows)
    N = sum(len(ws) for ws in docs_windows)
    cooc = count_cooccurrences(docs_windows, vocab)
    graph = build_weighted_graph(cooc, vocab, method="frequency")
    filtered = threshold_filter(graph, threshold=1.0)
    communities = connected_components(filtered)
    assignment = anchor_communities(communities, seeds)
    centrality = degree_centrality(filtered)
    ideology_terms = build_ideology_term_map(communities, assignment, centrality)
    return {
        "ideology_terms": ideology_terms,
        "graph_edges": [(u, v, w) for u, v, w in filtered.edges()],
    }


class TestIntegration:

    def test_empty_corpus_classify(self):
        """Corpus vazio → modelo sem termos → classificação uniforme."""
        seeds = {"A": ["mercado"], "B": ["saúde"]}
        model = {
            "ideology_terms": {"A": {}, "B": {}},
            "graph_edges": [],
        }
        terms = process_document("", use_lemmatizer=False)
        flat = [t for w in terms for t in w]
        scores = classify(flat, model, method="jaccard")
        total = sum(scores.values())
        assert total == pytest.approx(1.0, abs=1e-9)

    def test_all_stopwords_corpus(self):
        """Corpus de texto composto só de stopwords → sem janelas → modelo vazio."""
        docs = ["o e a de do para com em uma", "que se em ou mas"]
        docs_windows = process_corpus(docs, use_lemmatizer=False, window_size=3)
        assert all(len(ws) == 0 for ws in docs_windows)

    def test_single_word_corpus(self):
        """Corpus onde cada doc tem só 1 palavra de conteúdo → sem coocorrências."""
        docs = ["mercado", "saúde", "educação"]
        docs_windows = process_corpus(docs, use_lemmatizer=False, window_size=2)
        vocab = build_vocab_from_windows(docs_windows)
        cooc = count_cooccurrences(docs_windows, vocab)
        assert len(cooc) == 0

    def test_score_sums_to_one_always(self):
        """Para qualquer entrada, a soma das probabilidades deve ser 1."""
        docs = [
            "mercado livre gera riqueza prosperidade econômica eficiência capital",
            "saúde pública educação direitos trabalhadores sindicato reforma social",
            "família tradicional valores morais ordem segurança pátria nação",
        ]
        seeds = {
            "libertarianismo": ["mercado", "capital", "eficiência"],
            "social-democracia": ["saúde", "educação", "trabalhadores"],
            "conservadorismo": ["família", "valores", "pátria"],
        }
        model = _build_minimal_model(docs, seeds)
        test_docs = [
            "mercado privatização capital lucro",           # libertarianismo
            "saúde pública trabalhadores direitos greve",   # social-democracia
            "família tradição ordem segurança pátria",      # conservadorismo
            "texto completamente fora do vocabulário xyz",  # desconhecido
            "",                                             # vazio
        ]
        for doc in test_docs:
            windows = process_document(doc, use_lemmatizer=False, window_size=3)
            terms = list({t for w in windows for t in w})
            scores = classify(terms, model, method="jaccard")
            assert sum(scores.values()) == pytest.approx(1.0, abs=1e-9), \
                f"Falhou para doc: {doc!r}"