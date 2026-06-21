"""Métricas de centralidade: grau, intermediação (Brandes) e top-k via heap."""

from __future__ import annotations

import heapq
from collections import deque

from src.datastructures.graph import Graph


def degree_centrality(graph: Graph) -> dict[str, float]:
    """Centralidade de grau normalizada pelo número máximo de vizinhos possíveis.

    C_d(v) = grau(v) / (n - 1),  onde n = número de vértices.

    Args:
        graph: Grafo não-direcionado.

    Returns:
        Dicionário {vértice: centralidade ∈ [0, 1]}.
    """
    n = graph.num_vertices()
    if n <= 1:
        return {v: 0.0 for v in graph.vertices()}
    denom = n - 1
    return {v: graph.degree(v) / denom for v in graph.vertices()}


def betweenness_brandes(graph: Graph) -> dict[str, float]:
    """Centralidade de intermediação via algoritmo de Brandes (2001).

    Implementação O(VE) para grafos não-ponderados (pesos ignorados na
    BFS). Normalizada por 2 / ((n-1)(n-2)) para grafos não-direcionados.

    Args:
        graph: Grafo não-direcionado.

    Returns:
        Dicionário {vértice: centralidade de intermediação normalizada}.
    """
    vertices = graph.vertices()
    n = len(vertices)
    cb: dict[str, float] = {v: 0.0 for v in vertices}

    for s in vertices:
        # Pilha de vértices na ordem de descoberta (para acumulação reversa).
        stack: list[str] = []
        # Predecessores no caminho mais curto.
        pred: dict[str, list[str]] = {v: [] for v in vertices}
        # Número de caminhos mínimos de s a v.
        sigma: dict[str, float] = {v: 0.0 for v in vertices}
        sigma[s] = 1.0
        # Distância de s a v (-1 = não visitado).
        dist: dict[str, int] = {v: -1 for v in vertices}
        dist[s] = 0

        queue: deque[str] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.neighbors(v):
                # Descoberta de w pela primeira vez?
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                # É caminho mínimo para w via v?
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        # Acumulação reversa.
        delta: dict[str, float] = {v: 0.0 for v in vertices}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                cb[w] += delta[w]

    # Normalização para grafos não-direcionados.
    if n > 2:
        norm = 2.0 / ((n - 1) * (n - 2))
        cb = {v: cb[v] * norm for v in cb}

    return cb


def top_k(scores: dict[str, float], k: int) -> list[tuple[str, float]]:
    """Retorna os k vértices com maior pontuação usando uma min-heap de tamanho k.

    Args:
        scores: Dicionário {vértice: pontuação}.
        k: Número de itens a retornar.

    Returns:
        Lista de (vértice, pontuação) em ordem decrescente de pontuação.
    """
    if k <= 0:
        return []
    # Min-heap de tamanho k: (score, vertex)
    heap: list[tuple[float, str]] = []
    for v, s in scores.items():
        if len(heap) < k:
            heapq.heappush(heap, (s, v))
        elif s > heap[0][0]:
            heapq.heapreplace(heap, (s, v))
    return sorted(((v, s) for s, v in heap), key=lambda x: -x[1])
