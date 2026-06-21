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
