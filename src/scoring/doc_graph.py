"""Construção do grafo de co-ocorrência de um documento novo."""

from __future__ import annotations

from src.datastructures.graph import Graph


def build_doc_graph(windows: list[list[str]]) -> Graph:
    """Constrói um grafo de co-ocorrência a partir das janelas de um documento.

    Para cada janela, todos os pares não-ordenados de termos distintos recebem
    uma aresta. Janelas múltiplas com o mesmo par acumulam peso — termos que
    co-ocorrem em mais janelas formam arestas mais fortes.

    Args:
        windows: Janelas deslizantes do documento (saída de process_document).

    Returns:
        Grafo ponderado onde peso(u, v) = número de janelas em que u e v coocorrem.
    """
    graph = Graph()
    for window in windows:
        unique = list(set(window))
        n = len(unique)
        for i in range(n):
            for j in range(i + 1, n):
                graph.add_edge(unique[i], unique[j], 1.0)
    return graph
