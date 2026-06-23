"""Testes para analysis: filtering, traversal, communities, centrality."""

import pytest

from src.datastructures.graph import Graph
from src.analysis.traversal import bfs, connected_components
from src.analysis.filtering import max_spanning_backbone, threshold_filter, disparity_filter
from src.analysis.centrality import degree_centrality, betweenness_brandes, top_k
from src.analysis.communities import detect_communities, label_propagation


# ─────────────────────────────────────────────
# Grafo de apoio
# ─────────────────────────────────────────────

def make_triangle() -> Graph:
    """Triângulo a-b-c com pesos distintos."""
    g = Graph()
    g.set_edge("a", "b", 3.0)
    g.set_edge("b", "c", 1.0)
    g.set_edge("a", "c", 2.0)
    return g


def make_two_cliques() -> Graph:
    """Duas cliques de 3 vértices conectadas por uma ponte fraca."""
    g = Graph()
    # Clique 1: a, b, c
    g.set_edge("a", "b", 5.0)
    g.set_edge("b", "c", 5.0)
    g.set_edge("a", "c", 5.0)
    # Clique 2: x, y, z
    g.set_edge("x", "y", 5.0)
    g.set_edge("y", "z", 5.0)
    g.set_edge("x", "z", 5.0)
    # Ponte fraca entre as duas cliques
    g.set_edge("c", "x", 0.1)
    return g


# ─────────────────────────────────────────────
# Traversal
# ─────────────────────────────────────────────

class TestTraversal:
    def test_bfs_visits_all_in_connected(self):
        g = make_triangle()
        result = bfs(g, "a")
        assert sorted(result) == ["a", "b", "c"]

    def test_bfs_starts_with_source(self):
        g = make_triangle()
        assert bfs(g, "b")[0] == "b"

    def test_connected_components_two_groups(self):
        g = Graph()
        g.set_edge("a", "b", 1.0)
        g.set_edge("x", "y", 1.0)
        comps = connected_components(g)
        assert len(comps) == 2
        sizes = sorted(len(c) for c in comps)
        assert sizes == [2, 2]

    def test_connected_components_single_component(self):
        g = make_triangle()
        comps = connected_components(g)
        assert len(comps) == 1
        assert sorted(comps[0]) == ["a", "b", "c"]

    def test_connected_components_isolated_vertices(self):
        g = Graph()
        g.add_vertex("solo1")
        g.add_vertex("solo2")
        comps = connected_components(g)
        assert len(comps) == 2


# ─────────────────────────────────────────────
# Filtering
# ─────────────────────────────────────────────

class TestFiltering:
    def test_kruskal_is_spanning_tree(self):
        """Árvore geradora deve ter V-1 arestas para grafo conexo."""
        g = make_triangle()
        backbone = max_spanning_backbone(g)
        n = backbone.num_vertices()
        e = backbone.num_edges()
        assert e == n - 1

    def test_kruskal_keeps_heaviest_edges(self):
        """A aresta mais pesada (a-b, peso 3) deve estar na backbone."""
        g = make_triangle()
        backbone = max_spanning_backbone(g)
        assert backbone.has_edge("a", "b")

    def test_kruskal_forest_two_components(self):
        """Dois componentes → floresta com 2*(V/2-1) arestas."""
        g = Graph()
        g.set_edge("a", "b", 2.0)
        g.set_edge("x", "y", 1.0)
        backbone = max_spanning_backbone(g)
        # 4 vértices, 2 componentes → floresta com 2 arestas
        assert backbone.num_edges() == 2

    def test_threshold_filter_removes_low(self):
        g = make_triangle()
        filtered = threshold_filter(g, threshold=2.0)
        assert not filtered.has_edge("b", "c")  # peso 1.0 < 2.0
        assert filtered.has_edge("a", "b")       # peso 3.0 >= 2.0

    def test_threshold_filter_keeps_all(self):
        g = make_triangle()
        filtered = threshold_filter(g, threshold=0.0)
        assert filtered.num_edges() == g.num_edges()

    def test_disparity_filter_returns_graph(self):
        g = make_two_cliques()
        filtered = disparity_filter(g, alpha=0.1)
        assert isinstance(filtered, Graph)
        assert filtered.num_vertices() == g.num_vertices()
        
    def test_kruskal_empty_graph(self):
        """max_spanning_backbone de grafo vazio → grafo vazio."""
        from src.datastructures.graph import Graph
        from src.analysis.filtering import max_spanning_backbone
        g = Graph()
        result = max_spanning_backbone(g)
        assert result.num_vertices() == 0
        assert result.num_edges() == 0
    
    def test_kruskal_single_edge(self):
        """Grafo com 1 aresta: backbone tem a mesma aresta."""
        from src.datastructures.graph import Graph
        from src.analysis.filtering import max_spanning_backbone
        g = Graph()
        g.set_edge("a", "b", 1.0)
        result = max_spanning_backbone(g)
        assert result.has_edge("a", "b")
        assert result.num_edges() == 1
    
    def test_kruskal_isolated_vertices_preserved(self):
        """Vértices isolados devem aparecer no backbone (sem arestas)."""
        from src.datastructures.graph import Graph
        from src.analysis.filtering import max_spanning_backbone
        g = Graph()
        g.add_vertex("solo")
        g.set_edge("a", "b", 1.0)
        result = max_spanning_backbone(g)
        assert "solo" in result.vertices()
    
    def test_threshold_filter_empty_graph(self):
        """threshold_filter de grafo vazio → grafo vazio."""
        from src.datastructures.graph import Graph
        from src.analysis.filtering import threshold_filter
        g = Graph()
        result = threshold_filter(g, threshold=0.5)
        assert result.num_edges() == 0
    
    def test_threshold_filter_removes_all(self):
        """threshold muito alto → remove todas as arestas, mantém vértices."""
        from src.datastructures.graph import Graph
        from src.analysis.filtering import threshold_filter
        g = Graph()
        g.set_edge("a", "b", 0.1)
        result = threshold_filter(g, threshold=999.0)
        assert result.num_edges() == 0
        assert "a" in result.vertices()
    
    def test_disparity_filter_empty_graph(self):
        """disparity_filter de grafo vazio → grafo vazio."""
        from src.datastructures.graph import Graph
        from src.analysis.filtering import disparity_filter
        g = Graph()
        result = disparity_filter(g, alpha=0.05)
        assert result.num_edges() == 0
    
    def test_disparity_filter_degree_one_keeps_edge(self):
        """Vértice de grau 1: sua única aresta sempre é mantida."""
        from src.datastructures.graph import Graph
        from src.analysis.filtering import disparity_filter
        g = Graph()
        g.set_edge("a", "b", 1.0)  # ambos têm grau 1
        result = disparity_filter(g, alpha=0.05)
        assert result.has_edge("a", "b")



# ─────────────────────────────────────────────
# Centrality
# ─────────────────────────────────────────────

class TestCentrality:
    def test_degree_centrality_star(self):
        """Centro de estrela tem centralidade 1.0; folhas têm 1/(n-1)."""
        g = Graph()
        for i in range(4):
            g.set_edge("center", f"leaf{i}", 1.0)
        dc = degree_centrality(g)
        assert dc["center"] == pytest.approx(1.0)

    def test_degree_centrality_sums_correctly(self):
        g = make_triangle()
        dc = degree_centrality(g)
        for v, c in dc.items():
            assert 0.0 <= c <= 1.0

    def test_betweenness_brandes_path_graph(self):
        """No grafo linear a-b-c, b deve ter a maior intermediação."""
        g = Graph()
        g.set_edge("a", "b", 1.0)
        g.set_edge("b", "c", 1.0)
        cb = betweenness_brandes(g)
        assert cb["b"] > cb["a"]
        assert cb["b"] > cb["c"]

    def test_betweenness_triangle_uniform(self):
        """Triângulo: todos os vértices têm intermediação zero (caminhos diretos)."""
        g = make_triangle()
        cb = betweenness_brandes(g)
        for v, c in cb.items():
            assert c == pytest.approx(0.0, abs=1e-9)

    def test_top_k_returns_k_items(self):
        scores = {"a": 0.5, "b": 0.9, "c": 0.1, "d": 0.7}
        result = top_k(scores, k=2)
        assert len(result) == 2
        assert result[0][0] == "b"
        assert result[1][0] == "d"

    def test_top_k_zero_returns_empty(self):
        assert top_k({"a": 1.0}, k=0) == []
        
    def test_degree_centrality_empty_graph(self):
        """degree_centrality de grafo vazio → dicionário vazio."""
        from src.datastructures.graph import Graph
        from src.analysis.centrality import degree_centrality
        g = Graph()
        assert degree_centrality(g) == {}
    
    def test_degree_centrality_single_vertex(self):
        """Grafo com 1 vértice isolado → centralidade 0.0 (n-1 = 0, guarda divisão)."""
        from src.datastructures.graph import Graph
        from src.analysis.centrality import degree_centrality
        g = Graph()
        g.add_vertex("solo")
        dc = degree_centrality(g)
        assert dc["solo"] == 0.0
    
    def test_betweenness_empty_graph(self):
        """betweenness_brandes de grafo vazio → dicionário vazio."""
        from src.datastructures.graph import Graph
        from src.analysis.centrality import betweenness_brandes
        g = Graph()
        assert betweenness_brandes(g) == {}
    
    def test_betweenness_two_vertices(self):
        """2 vértices: intermediação de ambos deve ser 0.0 (n <= 2, sem normalização)."""
        from src.datastructures.graph import Graph
        from src.analysis.centrality import betweenness_brandes
        g = Graph()
        g.set_edge("a", "b", 1.0)
        cb = betweenness_brandes(g)
        assert cb["a"] == pytest.approx(0.0)
        assert cb["b"] == pytest.approx(0.0)
    
    def test_top_k_more_than_available(self):
        """k maior que o número de itens → retorna todos os itens disponíveis."""
        from src.analysis.centrality import top_k
        scores = {"a": 1.0, "b": 2.0}
        result = top_k(scores, k=100)
        assert len(result) == 2
    
    def test_top_k_empty_scores(self):
        """top_k com scores vazio → lista vazia."""
        from src.analysis.centrality import top_k
        assert top_k({}, k=5) == []
    
    def test_top_k_all_same_score(self):
        """top_k com todos os scores iguais → retorna k itens."""
        from src.analysis.centrality import top_k
        scores = {str(i): 1.0 for i in range(10)}
        result = top_k(scores, k=3)
        assert len(result) == 3


# ─────────────────────────────────────────────
# Communities
# ─────────────────────────────────────────────

class TestCommunities:
    def test_girvan_newman_splits_two_cliques(self):
        """Girvan-Newman deve separar as duas cliques conectadas por ponte fraca."""
        g = make_two_cliques()
        comps = detect_communities(g, max_communities=2)
        assert len(comps) == 2
        sizes = sorted(len(c) for c in comps)
        assert sizes == [3, 3]

    def test_girvan_newman_disconnected_already(self):
        g = Graph()
        g.set_edge("a", "b", 1.0)
        g.set_edge("x", "y", 1.0)
        comps = detect_communities(g, max_communities=2)
        assert len(comps) == 2

    def test_label_propagation_two_cliques(self):
        g = make_two_cliques()
        comps = label_propagation(g, seed=42)
        assert len(comps) >= 2

    def test_label_propagation_empty_graph(self):
        g = Graph()
        comps = label_propagation(g)
        assert comps == []

    def test_label_propagation_distribution_sums(self):
        """Todos os vértices devem aparecer em exatamente uma comunidade."""
        g = make_two_cliques()
        comps = label_propagation(g, seed=0)
        all_v = [v for c in comps for v in c]
        assert sorted(all_v) == sorted(g.vertices())
