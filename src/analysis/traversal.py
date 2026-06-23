"""Travessias de grafo: BFS e detecção de componentes conexos."""

from __future__ import annotations

from collections import deque

from src.datastructures.graph import Graph


def bfs(graph: Graph, start: str) -> list[str]:
    """Busca em largura (BFS) a partir de um vértice inicial.

    Implementação manual com fila explícita (deque) e conjunto de visitados.
    Não usa nenhuma função de travessia de biblioteca.

    Args:
        graph: Grafo não-direcionado.
        start: Vértice inicial.

    Returns:
        Lista de vértices na ordem de visita BFS.
    """
    visited: set[str] = {start}
    queue: deque[str] = deque([start])
    order: list[str] = []

    while queue:
        v = queue.popleft()
        order.append(v)
        for nbr in sorted(graph.neighbors(v)):  # ordem determinística
            if nbr not in visited:
                visited.add(nbr)
                queue.append(nbr)

    return order


def connected_components(graph: Graph) -> list[list[str]]:
    """Identifica todos os componentes conexos do grafo.

    Usa BFS a partir de vértices ainda não visitados.

    Args:
        graph: Grafo não-direcionado.

    Returns:
        Lista de componentes; cada componente é uma lista de vértices.
        Componentes são retornados em ordem decrescente de tamanho.
    """
    visited: set[str] = set()
    components: list[list[str]] = []

    for v in sorted(graph.vertices()):  # ordem determinística
        if v not in visited:
            comp = bfs(graph, v)
            components.append(comp)
            visited.update(comp)

    components.sort(key=len, reverse=True)
    return components
