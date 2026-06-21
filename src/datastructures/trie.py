"""Trie (árvore de prefixos) para casamento de expressões multipalavras."""

from __future__ import annotations

from typing import Optional


class _TrieNode:
    """Nó interno da Trie.

    Attributes:
        children: Mapa de token -> nó filho.
        terminal: Se True, este nó marca o fim de uma frase inserida.
        phrase: A frase completa armazenada no nó terminal (para recuperação).
    """

    __slots__ = ("children", "terminal", "phrase")

    def __init__(self) -> None:
        self.children: dict[str, "_TrieNode"] = {}
        self.terminal: bool = False
        self.phrase: Optional[str] = None


class Trie:
    """Trie para inserção e casamento de marcadores multipalavra.

    Cada frase é tokenizada por espaço e inserida na trie.
    O método match_longest encontra o marcador mais longo que inicia
    na posição i de uma lista de tokens.

    Exemplo:
        >>> t = Trie()
        >>> t.insert("livre mercado")
        >>> t.insert("livre")
        >>> t.match_longest(["livre", "mercado", "é"], 0)
        ('livre mercado', 2)
        >>> t.match_longest(["livre", "é"], 0)
        ('livre', 1)
        >>> t.match_longest(["bom", "dia"], 0)
        (None, 0)
    """

    def __init__(self) -> None:
        """Inicializa a Trie com a raiz vazia."""
        self._root = _TrieNode()

    def insert(self, phrase: str) -> None:
        """Insere uma frase (possivelmente multipalavra) na Trie.

        Args:
            phrase: Frase a inserir; tokens separados por espaço.
        """
        tokens = phrase.strip().split()
        node = self._root
        for tok in tokens:
            if tok not in node.children:
                node.children[tok] = _TrieNode()
            node = node.children[tok]
        node.terminal = True
        node.phrase = phrase

    def match_longest(
        self, tokens: list[str], i: int
    ) -> tuple[Optional[str], int]:
        """Casa o marcador mais longo que começa na posição i da lista de tokens.

        Args:
            tokens: Lista de tokens do documento.
            i: Posição inicial de busca.

        Returns:
            Tupla (frase_casada, próxima_posição). Se nenhum marcador casar,
            retorna (None, i).
        """
        node = self._root
        last_match: Optional[str] = None
        last_pos: int = i

        j = i
        while j < len(tokens) and tokens[j] in node.children:
            node = node.children[tokens[j]]
            j += 1
            if node.terminal:
                last_match = node.phrase
                last_pos = j

        return last_match, last_pos

    def contains(self, phrase: str) -> bool:
        """Verifica se uma frase exata está na Trie.

        Args:
            phrase: Frase a verificar.

        Returns:
            True se a frase foi inserida anteriormente.
        """
        tokens = phrase.strip().split()
        node = self._root
        for tok in tokens:
            if tok not in node.children:
                return False
            node = node.children[tok]
        return node.terminal
