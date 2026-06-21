"""Testes para as estruturas de dados: Graph, UnionFind e Trie."""

import pytest

from src.datastructures.graph import Graph
from src.datastructures.union_find import UnionFind
from src.datastructures.trie import Trie


# ─────────────────────────────────────────────
# UnionFind
# ─────────────────────────────────────────────

class TestUnionFind:
    def test_find_singleton_returns_itself(self):
        uf = UnionFind()
        assert uf.find("a") == "a"

    def test_union_connects_elements(self):
        uf = UnionFind()
        uf.union("a", "b")
        assert uf.connected("a", "b")

    def test_union_transitivity(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.connected("a", "c")

    def test_union_returns_false_when_already_connected(self):
        uf = UnionFind()
        uf.union("x", "y")
        assert uf.union("x", "y") is False

    def test_union_returns_true_on_new_merge(self):
        uf = UnionFind()
        assert uf.union("p", "q") is True

    def test_distinct_components(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        assert not uf.connected("a", "c")

    def test_path_compression(self):
        uf = UnionFind()
        for i in range(10):
            uf.union(i, i + 1)
        root = uf.find(0)
        # Após compressão, todos devem apontar para a raiz.
        for i in range(1, 11):
            assert uf.find(i) == root

    def test_components_groups_correctly(self):
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        comps = uf.components()
        sizes = sorted(len(v) for v in comps.values())
        assert sizes == [2, 2]

    def test_integer_keys(self):
        uf = UnionFind()
        uf.union(0, 1)
        uf.union(1, 2)
        assert uf.connected(0, 2)
        assert not uf.connected(0, 3)
    
    def test_self_union(self):
        """Union de um elemento consigo mesmo deve retornar False."""
        uf = UnionFind()
        result = uf.union("a", "a")
        assert result is False
    
    def test_find_new_element(self):
        """find de elemento novo cria-o como singleton e retorna ele mesmo."""
        uf = UnionFind()
        assert uf.find("x") == "x"
    
    def test_transitive_connectivity(self):
        """a-b e b-c → a e c devem estar conectados (transitividade)."""
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("b", "c")
        assert uf.connected("a", "c")
    
    def test_large_chain(self):
        """Cadeia de 1000 elementos: primeiro e último devem estar conectados."""
        uf = UnionFind()
        elements = [str(i) for i in range(1000)]
        for i in range(len(elements) - 1):
            uf.union(elements[i], elements[i + 1])
        assert uf.connected("0", "999")
    
    def test_disconnected_elements(self):
        """Elementos nunca unidos não devem estar conectados."""
        uf = UnionFind()
        uf.union("a", "b")
        assert not uf.connected("a", "c")
        assert not uf.connected("b", "c")
    
    def test_components_count(self):
        """3 grupos independentes → components deve retornar 3 entradas."""
        uf = UnionFind()
        uf.union("a", "b")
        uf.union("c", "d")
        uf.find("e")  # singleton
        comps = uf.components()
        assert len(comps) == 3


# ─────────────────────────────────────────────
# Graph
# ─────────────────────────────────────────────

class TestGraph:
    def test_add_edge_creates_vertices(self):
        g = Graph()
        g.add_edge("a", "b", 1.0)
        assert "a" in g.vertices()
        assert "b" in g.vertices()

    def test_degree(self):
        g = Graph()
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "c", 2.0)
        assert g.degree("a") == 2
        assert g.degree("b") == 1

    def test_weighted_degree(self):
        g = Graph()
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "c", 2.0)
        assert g.weighted_degree("a") == pytest.approx(3.0)

    def test_neighbors(self):
        g = Graph()
        g.add_edge("x", "y", 5.0)
        nbrs = g.neighbors("x")
        assert "y" in nbrs
        assert nbrs["y"] == pytest.approx(5.0)

    def test_add_edge_accumulates_weight(self):
        g = Graph()
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "b", 3.0)
        assert g.edge_weight("a", "b") == pytest.approx(4.0)

    def test_no_self_loops(self):
        g = Graph()
        g.add_edge("a", "a", 1.0)
        assert g.degree("a") == 0

    def test_edges_no_duplicates(self):
        g = Graph()
        g.add_edge("a", "b", 1.0)
        g.add_edge("b", "c", 2.0)
        assert g.num_edges() == 2

    def test_remove_edge(self):
        g = Graph()
        g.add_edge("a", "b", 1.0)
        g.remove_edge("a", "b")
        assert not g.has_edge("a", "b")
        assert not g.has_edge("b", "a")

    def test_set_edge_overwrites(self):
        g = Graph()
        g.add_edge("a", "b", 1.0)
        g.set_edge("a", "b", 9.9)
        assert g.edge_weight("a", "b") == pytest.approx(9.9)

    def test_undirected_symmetry(self):
        g = Graph()
        g.add_edge("u", "v", 3.0)
        assert g.edge_weight("u", "v") == g.edge_weight("v", "u")

    def test_copy_is_independent(self):
        g = Graph()
        g.add_edge("a", "b", 1.0)
        h = g.copy()
        h.add_edge("a", "c", 2.0)
        assert not g.has_edge("a", "c")

    def test_isolated_vertex(self):
        g = Graph()
        g.add_vertex("solo")
        assert g.degree("solo") == 0
        assert g.weighted_degree("solo") == 0.0
        
    def test_self_loop_ignored(self):
        """add_edge(v, v) não deve criar auto-laço."""
        g = Graph()
        g.add_edge("a", "a", 5.0)
        assert not g.has_edge("a", "a")
        # Vértice deve existir mesmo sem aresta
        assert "a" in g.vertices()
    
    def test_set_edge_self_loop(self):
        """set_edge(v, v) também não deve criar auto-laço."""
        g = Graph()
        g.set_edge("x", "x", 10.0)
        assert g.edge_weight("x", "x") == 0.0
    
    def test_add_edge_accumulates_weight(self):
        """Chamadas múltiplas a add_edge devem somar os pesos."""
        g = Graph()
        g.add_edge("a", "b", 1.0)
        g.add_edge("a", "b", 2.0)
        assert g.edge_weight("a", "b") == pytest.approx(3.0)
    
    def test_set_edge_overwrites(self):
        """set_edge deve substituir (não acumular) o peso anterior."""
        g = Graph()
        g.set_edge("a", "b", 5.0)
        g.set_edge("a", "b", 1.0)
        assert g.edge_weight("a", "b") == pytest.approx(1.0)
    
    def test_remove_edge_nonexistent(self):
        """Remover aresta inexistente não deve lançar exceção."""
        g = Graph()
        g.add_vertex("a")
        g.remove_edge("a", "b")  # b nem existe — não deve explodir
    
    def test_empty_graph_metrics(self):
        """Grafo vazio deve retornar zeros em todas as métricas."""
        g = Graph()
        assert g.num_vertices() == 0
        assert g.num_edges() == 0
        assert g.vertices() == []
        assert g.edges() == []
    
    def test_degree_isolated_vertex(self):
        """Vértice isolado tem grau zero e weighted_degree zero."""
        g = Graph()
        g.add_vertex("solo")
        assert g.degree("solo") == 0
        assert g.weighted_degree("solo") == 0.0
    
    def test_degree_unknown_vertex(self):
        """degree() de vértice que não existe deve retornar 0."""
        g = Graph()
        assert g.degree("fantasma") == 0
    
    def test_edge_symmetry(self):
        """Grafo não-direcionado: edge(u,v) == edge(v,u)."""
        g = Graph()
        g.set_edge("a", "b", 3.7)
        assert g.edge_weight("a", "b") == g.edge_weight("b", "a")
        assert g.has_edge("a", "b") == g.has_edge("b", "a")
    
    def test_copy_independence(self):
        """Copiar o grafo e modificar a cópia não afeta o original."""
        g = Graph()
        g.set_edge("a", "b", 1.0)
        g2 = g.copy()
        g2.set_edge("a", "b", 99.0)
        assert g.edge_weight("a", "b") == pytest.approx(1.0)
    
    def test_edges_no_duplicates(self):
        """edges() nunca deve retornar o mesmo par (u,v) duas vezes."""
        g = Graph()
        for u, v in [("a","b"), ("b","c"), ("a","c"), ("c","d")]:
            g.set_edge(u, v, 1.0)
        edges = g.edges()
        pairs = [frozenset((u, v)) for u, v, _ in edges]
        assert len(pairs) == len(set(pairs))
    
    def test_zero_weight_edge(self):
        """Aresta com peso 0.0 deve existir no grafo."""
        g = Graph()
        g.set_edge("a", "b", 0.0)
        assert g.has_edge("a", "b")
    
    def test_negative_weight_edge(self):
        """Aresta com peso negativo deve ser aceita pelo grafo (sem validação de sinal)."""
        g = Graph()
        g.set_edge("a", "b", -1.0)
        assert g.has_edge("a", "b")

        
    


# ─────────────────────────────────────────────
# Trie
# ─────────────────────────────────────────────

class TestTrie:
    def test_contains_after_insert(self):
        t = Trie()
        t.insert("livre mercado")
        assert t.contains("livre mercado")

    def test_not_contains_absent(self):
        t = Trie()
        t.insert("livre mercado")
        assert not t.contains("livre")

    def test_match_longest_multiword(self):
        t = Trie()
        t.insert("livre mercado")
        t.insert("livre")
        phrase, nxt = t.match_longest(["livre", "mercado", "é"], 0)
        assert phrase == "livre mercado"
        assert nxt == 2

    def test_match_longest_fallback_to_shorter(self):
        t = Trie()
        t.insert("livre mercado")
        t.insert("livre")
        phrase, nxt = t.match_longest(["livre", "é"], 0)
        assert phrase == "livre"
        assert nxt == 1

    def test_match_longest_no_match(self):
        t = Trie()
        t.insert("livre mercado")
        phrase, nxt = t.match_longest(["bom", "dia"], 0)
        assert phrase is None
        assert nxt == 0

    def test_match_at_offset(self):
        t = Trie()
        t.insert("estado mínimo")
        tokens = ["mais", "estado", "mínimo", "hoje"]
        phrase, nxt = t.match_longest(tokens, 1)
        assert phrase == "estado mínimo"
        assert nxt == 3

    def test_insert_single_token(self):
        t = Trie()
        t.insert("privatização")
        assert t.contains("privatização")
        phrase, nxt = t.match_longest(["privatização", "é"], 0)
        assert phrase == "privatização"
        assert nxt == 1

    def test_match_empty_tokens(self):
        t = Trie()
        t.insert("teste")
        phrase, nxt = t.match_longest([], 0)
        assert phrase is None
        assert nxt == 0
        
    def test_empty_string_insert(self):
        """Inserir string vazia não deve causar erro; contains deve retornar True."""
        t = Trie()
        t.insert("")
        # Comportamento esperado: a raiz fica marcada como terminal
        # (a frase vazia é uma sequência de 0 tokens)
        assert t.contains("")
    
    def test_match_longest_empty_tokens(self):
        """match_longest em lista vazia deve retornar (None, 0)."""
        t = Trie()
        t.insert("mercado livre")
        phrase, pos = t.match_longest([], 0)
        assert phrase is None
        assert pos == 0
    
    def test_match_longest_out_of_bounds_i(self):
        """i maior que len(tokens) não deve lançar exceção."""
        t = Trie()
        t.insert("x")
        phrase, pos = t.match_longest(["a", "b"], 5)
        assert phrase is None
    
    def test_prefix_not_matched_as_phrase(self):
        """Prefixo de uma frase inserida não deve casar como frase completa."""
        t = Trie()
        t.insert("livre mercado")
        # "livre" isolado não está inserido
        assert not t.contains("livre")
    
    def test_longest_match_wins(self):
        """Quando 'livre' e 'livre mercado' estão inseridos, 'livre mercado' ganha."""
        t = Trie()
        t.insert("livre")
        t.insert("livre mercado")
        phrase, nxt = t.match_longest(["livre", "mercado", "gera"], 0)
        assert phrase == "livre mercado"
        assert nxt == 2
    
    def test_single_word_phrase(self):
        """Frase de uma palavra deve ser inserida e casada corretamente."""
        t = Trie()
        t.insert("neoliberalismo")
        phrase, nxt = t.match_longest(["neoliberalismo", "avança"], 0)
        assert phrase == "neoliberalismo"
        assert nxt == 1
    
    def test_duplicate_insert(self):
        """Inserir a mesma frase duas vezes não deve duplicar ou corromper."""
        t = Trie()
        t.insert("reforma agrária")
        t.insert("reforma agrária")
        assert t.contains("reforma agrária")
        phrase, nxt = t.match_longest(["reforma", "agrária"], 0)
        assert phrase == "reforma agrária"
        assert nxt == 2
    
    def test_does_not_contain_unseen_phrase(self):
        """Trie vazia não deve conter nenhuma frase."""
        t = Trie()
        assert not t.contains("qualquer coisa")
    
    def test_overlapping_phrases(self):
        """Frases com prefixo em comum: cada uma casa no contexto correto."""
        t = Trie()
        t.insert("estado mínimo")
        t.insert("estado maior")
        p1, _ = t.match_longest(["estado", "mínimo", "forte"], 0)
        p2, _ = t.match_longest(["estado", "maior", "forte"], 0)
        assert p1 == "estado mínimo"
        assert p2 == "estado maior"
