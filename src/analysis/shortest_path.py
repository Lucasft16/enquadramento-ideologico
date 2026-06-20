"""Caminho mínimo por custo inverso ao peso: Dijkstra com heap."""

from __future__ import annotations

import heapq
import math

from src.datastructures.graph import Graph


def dijkstra(
    graph: Graph, source: str
) -> tuple[dict[str, float], dict[str, str | None]]:
    """Dijkstra sobre custo = 1/peso (arestas com maior peso são "mais curtas").

    Args:
        graph: Grafo ponderado não-direcionado.
        source: Vértice de origem.

    Returns:
        Tupla (dist, prev) onde:
          dist[v]  = custo mínimo acumulado de source até v (∞ se inalcançável).
          prev[v]  = antecessor de v no caminho mínimo (None para source).
    """
    dist: dict[str, float] = {v: math.inf for v in graph.vertices()}
    prev: dict[str, str | None] = {v: None for v in graph.vertices()}

    if source not in dist:
        return dist, prev

    dist[source] = 0.0
    # Heap: (custo_acumulado, vértice)
    heap: list[tuple[float, str]] = [(0.0, source)]

    while heap:
        d, u = heapq.heappop(heap)
        if d > dist[u]:
            continue  # entrada obsoleta
        for v, w in graph.neighbors(u).items():
            cost = 1.0 / w if w > 0 else math.inf
            new_dist = dist[u] + cost
            if new_dist < dist[v]:
                dist[v] = new_dist
                prev[v] = u
                heapq.heappush(heap, (new_dist, v))

    return dist, prev


def reconstruct_path(prev: dict[str, str | None], target: str) -> list[str]:
    """Reconstrói o caminho mínimo de source até target usando o dicionário prev.

    Args:
        prev: Mapa vértice → antecessor retornado por dijkstra.
        target: Vértice de destino.

    Returns:
        Lista de vértices do caminho (source … target), ou lista vazia se
        target for inalcançável.
    """
    path: list[str] = []
    v: str | None = target
    while v is not None:
        path.append(v)
        v = prev.get(v)
    path.reverse()
    # Caminho válido só se começa em source (prev[source] == None).
    if len(path) == 1 and prev.get(target) is None and path[0] != target:
        return []
    return path
