"""Detecção de comunidades: Girvan-Newman e propagação de rótulos."""

from __future__ import annotations

import random
from collections import deque

from src.datastructures.graph import Graph
from src.analysis.traversal import connected_components
from src.analysis.centrality import betweenness_brandes


def _edge_betweenness(graph: Graph) -> dict[tuple[str, str], float]:
    """Calcula a intermediação de arestas via adaptação de Brandes.

    Para cada vértice s, executa BFS e distribui crédito às arestas
    nos caminhos mínimos encontrados.

    Args:
        graph: Grafo não-direcionado.

    Returns:
        Dicionário {(u, v): intermediação} com u <= v lexicograficamente.
    """
    vertices = graph.vertices()
    eb: dict[tuple[str, str], float] = {}
    for e in graph.edges():
        key = (e[0], e[1])
        eb[key] = 0.0

    for s in vertices:
        stack: list[str] = []
        pred: dict[str, list[str]] = {v: [] for v in vertices}
        sigma: dict[str, float] = {v: 0.0 for v in vertices}
        sigma[s] = 1.0
        dist: dict[str, int] = {v: -1 for v in vertices}
        dist[s] = 0
        queue: deque[str] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.neighbors(v):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta: dict[str, float] = {v: 0.0 for v in vertices}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                contribution = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                delta[v] += contribution
                # Distribui crédito à aresta (v, w).
                a, b = (v, w) if v <= w else (w, v)
                if (a, b) in eb:
                    eb[(a, b)] += contribution
            # delta[w] já foi acumulado nas iterações anteriores.

    # Normaliza (cada aresta foi contada duas vezes em grafos não-dir.)
    return {k: v / 2.0 for k, v in eb.items()}


def detect_communities(
    graph: Graph,
    max_communities: int = 10,
) -> list[list[str]]:
    """Girvan-Newman: remove iterativamente a aresta de maior intermediação.

    A cada iteração, recalcula a intermediação de arestas e remove a de
    maior valor. Para quando o grafo atinge `max_communities` componentes
    ou não há mais arestas.

    Args:
        graph: Grafo ponderado não-direcionado (cópia interna é usada).
        max_communities: Número máximo de comunidades desejado.

    Returns:
        Lista de comunidades; cada comunidade é uma lista de vértices.
    """
    g = graph.copy()

    while True:
        comps = connected_components(g)
        if len(comps) >= max_communities:
            break
        if g.num_edges() == 0:
            break

        eb = _edge_betweenness(g)
        if not eb:
            break

        # Remove a aresta com maior intermediação.
        best_edge = max(eb, key=lambda e: eb[e])
        g.remove_edge(best_edge[0], best_edge[1])

    return connected_components(g)


def label_propagation(
    graph: Graph,
    max_iter: int = 100,
    seed: int = 42,
) -> list[list[str]]:
    """Detecção de comunidades por propagação de rótulos (assíncrona).

    Cada vértice adota o rótulo mais frequente entre seus vizinhos.
    Convergência quando nenhum rótulo muda ou após `max_iter` iterações.

    Args:
        graph: Grafo ponderado não-direcionado.
        max_iter: Número máximo de iterações.
        seed: Semente para reprodutibilidade.

    Returns:
        Lista de comunidades detectadas.
    """
    rng = random.Random(seed)
    vertices = graph.vertices()
    if not vertices:
        return []

    # Inicializa cada vértice com seu próprio rótulo.
    labels: dict[str, str] = {v: v for v in vertices}

    for _ in range(max_iter):
        order = list(vertices)
        rng.shuffle(order)
        changed = False

        for v in order:
            nbrs = graph.neighbors(v)
            if not nbrs:
                continue

            # Conta rótulos dos vizinhos ponderados pelo peso da aresta.
            label_scores: dict[str, float] = {}
            for u, w in nbrs.items():
                lbl = labels[u]
                label_scores[lbl] = label_scores.get(lbl, 0.0) + w

            best = max(label_scores, key=lambda l: (label_scores[l], l))
            if best != labels[v]:
                labels[v] = best
                changed = True

        if not changed:
            break

    # Agrupa vértices pelo rótulo final.
    groups: dict[str, list[str]] = {}
    for v, lbl in labels.items():
        groups.setdefault(lbl, []).append(v)

    result = list(groups.values())
    result.sort(key=len, reverse=True)
    return result
