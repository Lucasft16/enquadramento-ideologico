"""Estrutura Union-Find (Disjoint Set Union) com compressão de caminho e união por rank."""

from __future__ import annotations

from typing import Hashable, TypeVar

T = TypeVar("T", bound=Hashable)


class UnionFind:
    """Conjunto disjunto genérico com compressão de caminho e união por rank.

    Suporta qualquer tipo hashable como elemento.

    Exemplo:
        >>> uf = UnionFind()
        >>> uf.union("a", "b")
        >>> uf.union("b", "c")
        >>> uf.find("a") == uf.find("c")
        True
    """

    def __init__(self) -> None:
        """Inicializa a estrutura vazia."""
        self._parent: dict[Hashable, Hashable] = {}
        self._rank: dict[Hashable, int] = {}

    def _make(self, x: Hashable) -> None:
        """Garante que x seja registrado como seu próprio pai (singleton)."""
        if x not in self._parent:
            self._parent[x] = x
            self._rank[x] = 0

    def find(self, x: Hashable) -> Hashable:
        """Retorna o representante do conjunto de x, com compressão de caminho.

        Args:
            x: Elemento a localizar.

        Returns:
            Representante (raiz) do conjunto que contém x.
        """
        self._make(x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])  # compressão de caminho
        return self._parent[x]

    def union(self, a: Hashable, b: Hashable) -> bool:
        """Une os conjuntos de a e b por rank.

        Args:
            a: Primeiro elemento.
            b: Segundo elemento.

        Returns:
            True se a união foi realizada (estavam em conjuntos distintos),
            False se já pertenciam ao mesmo conjunto.
        """
        self._make(a)
        self._make(b)
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return False
        if self._rank[ra] < self._rank[rb]:
            ra, rb = rb, ra
        self._parent[rb] = ra
        if self._rank[ra] == self._rank[rb]:
            self._rank[ra] += 1
        return True

    def connected(self, a: Hashable, b: Hashable) -> bool:
        """Verifica se a e b pertencem ao mesmo conjunto.

        Args:
            a: Primeiro elemento.
            b: Segundo elemento.

        Returns:
            True se estão no mesmo conjunto.
        """
        return self.find(a) == self.find(b)

    def components(self) -> dict[Hashable, list[Hashable]]:
        """Retorna um dicionário {representante: [membros]} para todos os elementos.

        Returns:
            Mapeamento de representante para lista de membros do componente.
        """
        groups: dict[Hashable, list[Hashable]] = {}
        for x in self._parent:
            root = self.find(x)
            groups.setdefault(root, []).append(x)
        return groups
