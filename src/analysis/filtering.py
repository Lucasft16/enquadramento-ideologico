"""Filtragem de grafos: Kruskal (backbone de peso máximo), threshold e disparity filter."""

from __future__ import annotations

from src.datastructures.graph import Graph
from src.datastructures.union_find import UnionFind


def max_spanning_backbone(graph: Graph) -> Graph:
    """Extrai a árvore/floresta geradora de peso máximo via algoritmo de Kruskal.

    As arestas são ordenadas em ordem decrescente de peso; uma aresta é
    incluída se não formar ciclo (usando UnionFind).

    Args:
        graph: Grafo ponderado de entrada.

    Returns:
        Novo grafo com apenas as arestas da floresta geradora de peso máximo.
    """
    edges = sorted(graph.edges(), key=lambda e: e[2], reverse=True)
    uf = UnionFind()
    backbone = Graph()

    for v in graph.vertices():
        backbone.add_vertex(v)

    for u, v, w in edges:
        if uf.union(u, v):  # sem ciclo → inclui a aresta
            backbone.set_edge(u, v, w)

    return backbone


def threshold_filter(graph: Graph, threshold: float) -> Graph:
    """Remove arestas com peso abaixo de um limiar.

    Args:
        graph: Grafo ponderado de entrada.
        threshold: Peso mínimo para manter a aresta (inclusivo).

    Returns:
        Novo grafo apenas com arestas de peso >= threshold.
    """
    filtered = Graph()
    for v in graph.vertices():
        filtered.add_vertex(v)
    for u, v, w in graph.edges():
        if w >= threshold:
            filtered.set_edge(u, v, w)
    return filtered


def disparity_filter(graph: Graph, alpha: float = 0.05) -> Graph:
    """Disparity filter: mantém arestas estatisticamente significativas por vértice.

    Para cada vértice v com grau k e força total s(v), uma aresta (v, u) com
    peso w é mantida se a probabilidade de uma aresta aleatória ter esse peso
    — sob distribuição uniforme — for menor que alpha:

        p(w) = (1 - w/s(v))^(k-1) < alpha

    A aresta é mantida se *qualquer* dos dois extremos a considera significativa.

    Referência: Serrano et al. (2009) PNAS.

    Args:
        graph: Grafo ponderado de entrada.
        alpha: Nível de significância (padrão 0.05).

    Returns:
        Novo grafo com arestas filtradas.
    """
    significant: set[frozenset[str]] = set()

    for v in graph.vertices():
        nbrs = graph.neighbors(v)
        k = len(nbrs)
        if k <= 1:
            # Vértice isolado ou grau 1: mantém a única aresta.
            for u in nbrs:
                significant.add(frozenset((v, u)))
            continue
        s = sum(nbrs.values())
        if s == 0:
            continue
        for u, w in nbrs.items():
            p = (1.0 - w / s) ** (k - 1)
            if p < alpha:
                significant.add(frozenset((v, u)))

    filtered = Graph()
    for v in graph.vertices():
        filtered.add_vertex(v)
    for u, v, w in graph.edges():
        if frozenset((u, v)) in significant:
            filtered.set_edge(u, v, w)
    return filtered
