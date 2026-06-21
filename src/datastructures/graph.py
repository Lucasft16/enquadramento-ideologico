"""Grafo ponderado não-direcionado implementado com lista de adjacência."""

from __future__ import annotations

from typing import Iterable, Iterator


class Graph:
    """Grafo ponderado não-direcionado representado por lista de adjacência.

    Vértices são identificados por strings (termos do vocabulário).
    Arestas armazenam um peso float. Múltiplas chamadas a add_edge com o
    mesmo par (u, v) somam os pesos (comportamento de acumulação de coocorrências).

    Exemplo:
        >>> g = Graph()
        >>> g.add_edge("a", "b", 1.0)
        >>> g.add_edge("b", "c", 2.5)
        >>> g.degree("b")
        2
        >>> g.weighted_degree("b")
        3.5
    """

    def __init__(self) -> None:
        """Inicializa grafo vazio."""
        # _adj[u][v] = peso da aresta (u, v)
        self._adj: dict[str, dict[str, float]] = {}

    # ------------------------------------------------------------------
    # Mutação
    # ------------------------------------------------------------------

    def add_vertex(self, v: str) -> None:
        """Garante que o vértice v existe no grafo (sem arestas).

        Args:
            v: Identificador do vértice.
        """
        if v not in self._adj:
            self._adj[v] = {}

    def add_edge(self, u: str, v: str, w: float = 1.0) -> None:
        """Adiciona (ou acumula peso em) uma aresta entre u e v.

        Args:
            u: Vértice de origem.
            v: Vértice de destino.
            w: Peso a acumular na aresta.
        """
        self.add_vertex(u)
        self.add_vertex(v)
        if u == v:
            return  # sem auto-laços
        self._adj[u][v] = self._adj[u].get(v, 0.0) + w
        self._adj[v][u] = self._adj[v].get(u, 0.0) + w

    def remove_edge(self, u: str, v: str) -> None:
        """Remove a aresta entre u e v, se existir.

        Args:
            u: Vértice de origem.
            v: Vértice de destino.
        """
        self._adj.get(u, {}).pop(v, None)
        self._adj.get(v, {}).pop(u, None)

    def set_edge(self, u: str, v: str, w: float) -> None:
        """Define (substitui) o peso de uma aresta.

        Útil após a etapa de ponderação para substituir contagens brutas por NPMI/Jaccard.

        Args:
            u: Vértice de origem.
            v: Vértice de destino.
            w: Novo peso.
        """
        self.add_vertex(u)
        self.add_vertex(v)
        if u == v:
            return  # sem auto-laços
        self._adj[u][v] = w
        self._adj[v][u] = w

    # ------------------------------------------------------------------
    # Consulta
    # ------------------------------------------------------------------

    def vertices(self) -> list[str]:
        """Retorna a lista de todos os vértices.

        Returns:
            Lista de identificadores de vértices.
        """
        return list(self._adj.keys())

    def edges(self) -> list[tuple[str, str, float]]:
        """Retorna todas as arestas como (u, v, peso), sem duplicatas.

        Returns:
            Lista de tuplas (u, v, peso) onde u < v lexicograficamente.
        """
        seen: set[frozenset[str]] = set()
        result: list[tuple[str, str, float]] = []
        for u, nbrs in self._adj.items():
            for v, w in nbrs.items():
                key = frozenset((u, v))
                if key not in seen:
                    seen.add(key)
                    a, b = (u, v) if u <= v else (v, u)
                    result.append((a, b, w))
        return result

    def neighbors(self, v: str) -> dict[str, float]:
        """Retorna dicionário {vizinho: peso} do vértice v.

        Args:
            v: Vértice consultado.

        Returns:
            Dicionário de vizinhos com seus pesos.
        """
        return dict(self._adj.get(v, {}))

    def degree(self, v: str) -> int:
        """Retorna o grau (número de vizinhos) do vértice v.

        Args:
            v: Vértice consultado.

        Returns:
            Número de arestas incidentes.
        """
        return len(self._adj.get(v, {}))

    def weighted_degree(self, v: str) -> float:
        """Retorna a soma dos pesos das arestas incidentes em v.

        Args:
            v: Vértice consultado.

        Returns:
            Soma dos pesos das arestas.
        """
        return sum(self._adj.get(v, {}).values())

    def has_edge(self, u: str, v: str) -> bool:
        """Verifica se existe aresta entre u e v.

        Args:
            u: Vértice de origem.
            v: Vértice de destino.

        Returns:
            True se a aresta existe.
        """
        return v in self._adj.get(u, {})

    def edge_weight(self, u: str, v: str) -> float:
        """Retorna o peso da aresta (u, v).

        Args:
            u: Vértice de origem.
            v: Vértice de destino.

        Returns:
            Peso da aresta, ou 0.0 se não existir.
        """
        return self._adj.get(u, {}).get(v, 0.0)

    def num_vertices(self) -> int:
        """Retorna o número de vértices.

        Returns:
            Contagem de vértices.
        """
        return len(self._adj)

    def num_edges(self) -> int:
        """Retorna o número de arestas (sem duplicatas).

        Returns:
            Contagem de arestas únicas.
        """
        return len(self.edges())

    def copy(self) -> "Graph":
        """Retorna uma cópia rasa do grafo (mesmos dados, novo objeto).

        Returns:
            Nova instância de Graph com as mesmas arestas e pesos.
        """
        g = Graph()
        for u, nbrs in self._adj.items():
            g._adj[u] = dict(nbrs)
        return g

    def __repr__(self) -> str:
        return f"Graph(V={self.num_vertices()}, E={self.num_edges()})"
