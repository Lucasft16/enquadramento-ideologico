"""Testes para graph_build: Vocabulary, coocorrência e ponderação (NPMI caso numérico)."""

import math
import pytest

from src.graph_build.vocabulary import Vocabulary
from src.graph_build.cooccurrence import count_cooccurrences, build_vocab_from_windows
from src.graph_build.weighting import build_weighted_graph, npmi, jaccard
from src.datastructures.graph import Graph


# ─────────────────────────────────────────────
# Vocabulary
# ─────────────────────────────────────────────

class TestVocabulary:
    def test_add_new_term(self):
        v = Vocabulary()
        tid = v.add("mercado")
        assert tid == 0
        assert v.term_id("mercado") == 0

    def test_add_idempotent(self):
        v = Vocabulary()
        t1 = v.add("estado")
        t2 = v.add("estado")
        assert t1 == t2

    def test_id_term_roundtrip(self):
        v = Vocabulary()
        v.add("liberdade")
        assert v.id_term(v.term_id("liberdade")) == "liberdade"

    def test_register_occurrence_increments_df(self):
        v = Vocabulary()
        v.register_occurrence("x")
        v.register_occurrence("x")
        assert v.document_frequency("x") == 2

    def test_document_frequency_unknown_is_zero(self):
        v = Vocabulary()
        assert v.document_frequency("naoexiste") == 0

    def test_len(self):
        v = Vocabulary()
        v.add("a")
        v.add("b")
        assert len(v) == 2

    def test_contains(self):
        v = Vocabulary()
        v.add("foo")
        assert "foo" in v
        assert "bar" not in v

    def test_prune_removes_low_frequency(self):
        v = Vocabulary()
        v.register_occurrence("raro")       # df=1
        v.register_occurrence("comum")
        v.register_occurrence("comum")      # df=2
        pruned = v.prune(min_df=2)
        assert "raro" not in pruned
        assert "comum" in pruned

    def test_prune_keeps_all_by_default(self):
        v = Vocabulary()
        for term in ["a", "b", "c"]:
            v.register_occurrence(term)
        pruned = v.prune(min_df=1, max_df_ratio=1.0)
        assert len(pruned) == 3

    def test_term_id_raises_for_unknown(self):
        """term_id de termo não registrado deve lançar KeyError."""
        v = Vocabulary()
        with pytest.raises(KeyError):
            v.term_id("fantasma")
    
    def test_id_term_raises_for_unknown(self):
        """id_term de ID inexistente deve lançar KeyError."""
        v = Vocabulary()
        with pytest.raises(KeyError):
            v.id_term(999)
    
    def test_prune_empty_vocabulary(self):
        """prune em vocabulário vazio deve retornar vocabulário vazio."""
        v = Vocabulary()
        pruned = v.prune(min_df=1)
        assert len(pruned) == 0
    
    def test_prune_max_df_ratio_zero(self):
        """max_df_ratio=0.0 → nenhum termo deve sobrar (max_abs = 0)."""
        v = Vocabulary()
        v.register_occurrence("comum")
        pruned = v.prune(min_df=1, max_df_ratio=0.0)
        assert len(pruned) == 0
    
    def test_vocabulary_terms_list(self):
        """terms() deve retornar exatamente os termos adicionados."""
        v = Vocabulary()
        for t in ["alfa", "beta", "gama"]:
            v.add(t)
        assert sorted(v.terms()) == ["alfa", "beta", "gama"]
    
    def test_iteration_order_preserved(self):
        """__iter__ deve iterar pelos termos na ordem de inserção."""
        v = Vocabulary()
        terms = ["z", "a", "m", "b"]
        for t in terms:
            v.add(t)
        assert list(v) == terms


# ─────────────────────────────────────────────
# Coocorrências
# ─────────────────────────────────────────────

class TestCooccurrence:
    def _make_vocab(self, terms):
        v = Vocabulary()
        for t in terms:
            v.add(t)
            v.register_occurrence(t)
        return v

    def test_count_single_window(self):
        vocab = self._make_vocab(["a", "b", "c"])
        windows = [[["a", "b", "c"]]]
        cooc = count_cooccurrences(windows, vocab)
        # 3 pares: (a,b), (a,c), (b,c)
        assert len(cooc) == 3

    def test_count_accumulates_across_windows(self):
        vocab = self._make_vocab(["x", "y"])
        # x e y aparecem juntos em 3 janelas
        windows = [[["x", "y"], ["x", "y"], ["x", "y"]]]
        cooc = count_cooccurrences(windows, vocab)
        key = (vocab.term_id("x"), vocab.term_id("y"))
        if key[0] > key[1]:
            key = (key[1], key[0])
        assert cooc[key] == 3

    def test_count_no_self_pairs(self):
        vocab = self._make_vocab(["z"])
        windows = [[["z", "z"]]]
        cooc = count_cooccurrences(windows, vocab)
        # Sem auto-pares
        assert len(cooc) == 0

    def test_build_vocab_from_windows(self):
        windows = [[["estado", "mercado"], ["mercado", "liberdade"]]]
        vocab = build_vocab_from_windows(windows)
        assert "estado" in vocab
        assert vocab.document_frequency("mercado") == 2  # aparece nas 2 janelas
        
    def test_count_cooccurrences_empty_corpus(self):
        """Corpus vazio → dicionário de coocorrências vazio."""
        vocab = Vocabulary()
        result = count_cooccurrences([], vocab)
        assert result == {}
    
    def test_count_cooccurrences_empty_windows(self):
        """Documento com lista de janelas vazia → sem coocorrências."""
        vocab = Vocabulary()
        vocab.add("a")
        result = count_cooccurrences([[]], vocab)
        assert result == {}
    
    def test_count_terms_not_in_vocab_ignored(self):
        """Termos não registrados no vocabulário devem ser ignorados silenciosamente."""
        vocab = Vocabulary()
        vocab.add("a")  # só "a" no vocab; "b" não está
        windows = [[["a", "b"]]]
        cooc = count_cooccurrences(windows, vocab)
        # Sem par válido (b não está no vocab)
        assert len(cooc) == 0
    
    def test_count_single_token_window(self):
        """Janela com apenas 1 token distinto → sem pares."""
        vocab = Vocabulary()
        vocab.add("x")
        windows = [[["x", "x", "x"]]]  # mesmo token repetido
        cooc = count_cooccurrences(windows, vocab)
        assert len(cooc) == 0



# ─────────────────────────────────────────────
# Ponderação – NPMI (caso numérico verificável)
# ─────────────────────────────────────────────

class TestWeighting:
    """
    Caso controlado:
      N = 100 janelas
      df(a) = 30  → P(a) = 0.30
      df(b) = 40  → P(b) = 0.40
      cooc(a,b) = 20 → P(a,b) = 0.20

      PMI  = log(0.20 / (0.30 * 0.40)) = log(0.20/0.12) = log(1.6667) ≈ 0.5108
      denom = -log(0.20)                                               ≈ 1.6094
      NPMI = 0.5108 / 1.6094                                          ≈ 0.3174
    """

    def _setup(self):
        vocab = Vocabulary()
        for t in ["a", "b"]:
            vocab.add(t)
        # Manualmente define frequências de documento
        vocab._doc_freq[vocab.term_id("a")] = 30
        vocab._doc_freq[vocab.term_id("b")] = 40
        return vocab

    def test_npmi_known_value(self):
        vocab = self._setup()
        ta, tb = vocab.term_id("a"), vocab.term_id("b")
        key = (ta, tb) if ta < tb else (tb, ta)
        cooc = {key: 20}
        g = Graph()
        npmi(cooc, vocab, g, N=100)
        w = g.edge_weight("a", "b")
        assert w == pytest.approx(0.3174, abs=1e-3)

    def test_npmi_zero_for_negative_pmi(self):
        """Quando cooc é muito baixa, NPMI negativo → sem aresta."""
        vocab = self._setup()
        ta, tb = vocab.term_id("a"), vocab.term_id("b")
        key = (ta, tb) if ta < tb else (tb, ta)
        # cooc = 1 → P(a,b) = 0.01 << P(a)*P(b) = 0.12 → PMI < 0
        cooc = {key: 1}
        g = Graph()
        npmi(cooc, vocab, g, N=100)
        assert not g.has_edge("a", "b")

    def test_frequency_method(self):
        vocab = Vocabulary()
        for t in ["p", "q"]:
            vocab.add(t)
        tp, tq = vocab.term_id("p"), vocab.term_id("q")
        key = (tp, tq) if tp < tq else (tq, tp)
        cooc = {key: 7}
        g = build_weighted_graph(cooc, vocab, method="frequency")
        assert g.edge_weight("p", "q") == pytest.approx(7.0)

    def test_jaccard_method(self):
        vocab = Vocabulary()
        for t in ["m", "n"]:
            vocab.add(t)
            vocab.register_occurrence(t)
        vocab.register_occurrence("m")  # df(m)=2, df(n)=1
        tm, tn = vocab.term_id("m"), vocab.term_id("n")
        key = (tm, tn) if tm < tn else (tn, tm)
        cooc = {key: 1}
        g = build_weighted_graph(cooc, vocab, method="jaccard")
        # J = 1 / (2 + 1 - 1) = 1/2 = 0.5
        assert g.edge_weight("m", "n") == pytest.approx(0.5)

    def test_unknown_method_raises(self):
        vocab = Vocabulary()
        with pytest.raises(ValueError):
            build_weighted_graph({}, vocab, method="invalido")

    def test_npmi_between_0_and_1(self):
        """Todos os pesos NPMI devem estar em [0, 1]."""
        vocab = Vocabulary()
        terms = ["x", "y", "z"]
        for t in terms:
            vocab.add(t)
        vocab._doc_freq[vocab.term_id("x")] = 50
        vocab._doc_freq[vocab.term_id("y")] = 60
        vocab._doc_freq[vocab.term_id("z")] = 10
        tx, ty, tz = [vocab.term_id(t) for t in terms]
        cooc = {
            (tx, ty) if tx < ty else (ty, tx): 40,
            (tx, tz) if tx < tz else (tz, tx): 8,
        }
        g = build_weighted_graph(cooc, vocab, method="npmi", N=100)
        for _, _, w in g.edges():
            assert 0.0 <= w <= 1.0 + 1e-9
            
    def test_npmi_n_zero_returns_empty_graph(self):
        """npmi com N=0 deve retornar grafo sem arestas (guarda divisão por zero)."""
        vocab = Vocabulary()
        vocab.add("a")
        vocab.add("b")
        g = Graph()
        result = npmi({}, vocab, g, N=0)
        assert result.num_edges() == 0
    
    def test_npmi_cooc_equals_n_p_ab_1(self):
        """cooc = N → P(a,b)=1 → -log(P(a,b))=0 → denominador=0 → aresta ignorada."""
        vocab = Vocabulary()
        vocab.add("a")
        vocab.add("b")
        vocab._doc_freq[0] = 10
        vocab._doc_freq[1] = 10
        # cooc = N=10 → p_ab = 1.0 → denom = -log(1) = 0 → ignorado
        key = (0, 1)
        g = Graph()
        result = npmi({key: 10}, vocab, g, N=10)
        assert not result.has_edge("a", "b")
    
    def test_jaccard_cooc_greater_than_df(self):
        """
        Se cooc > min(df_a, df_b), Jaccard pode ficar > 1 ou o denominador < 0.
        Documenta que o denominador <= 0 → aresta ignorada.
        """
        vocab = Vocabulary()
        vocab.add("a")
        vocab.add("b")
        # df(a)=1, df(b)=1, cooc=5 → denom = 1+1-5 = -3 <= 0 → ignorada
        vocab._doc_freq[vocab.term_id("a")] = 1
        vocab._doc_freq[vocab.term_id("b")] = 1
        ta, tb = vocab.term_id("a"), vocab.term_id("b")
        key = (ta, tb) if ta < tb else (tb, ta)
        g = Graph()
        jaccard({key: 5}, vocab, g)
        assert not g.has_edge("a", "b")
    
    def test_build_weighted_graph_empty_cooc(self):
        """Coocorrência vazia → grafo sem arestas para qualquer método."""
        for method in ["frequency", "npmi", "jaccard"]:
            vocab = Vocabulary()
            g = build_weighted_graph({}, vocab, method=method, N=10)
            assert g.num_edges() == 0
