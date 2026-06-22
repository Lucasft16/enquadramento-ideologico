"""Vocabulário: mapeia termos para IDs inteiros e rastreia frequências de documento."""

from __future__ import annotations

from typing import Iterator


class Vocabulary:
    """Vocabulário controlado para o grafo de coocorrência.

    Mantém dois mapeamentos (termo→id e id→termo) e a frequência de documento
    de cada termo (número de janelas distintas em que o termo apareceu).

    Exemplo:
        >>> vocab = Vocabulary()
        >>> vocab.add("estado")
        0
        >>> vocab.add("mercado")
        1
        >>> vocab.term_id("estado")
        0
    """

    def __init__(self) -> None:
        """Inicializa vocabulário vazio."""
        self._term_to_id: dict[str, int] = {}
        self._id_to_term: dict[int, str] = {}
        self._doc_freq: dict[int, int] = {}
        self._next_id: int = 0

    def add(self, term: str) -> int:
        """Adiciona um termo ao vocabulário, retornando seu ID.

        Se o termo já existir, apenas retorna o ID existente sem modificar
        a frequência de documento (use `register_occurrence` para isso).

        Args:
            term: Termo a registrar.

        Returns:
            ID inteiro do termo.
        """
        if term not in self._term_to_id:
            tid = self._next_id
            self._term_to_id[term] = tid
            self._id_to_term[tid] = term
            self._doc_freq[tid] = 0
            self._next_id += 1
        return self._term_to_id[term]

    def register_occurrence(self, term: str) -> int:
        """Adiciona o termo e incrementa sua frequência de documento.

        Args:
            term: Termo que ocorreu em uma janela.

        Returns:
            ID do termo.
        """
        tid = self.add(term)
        self._doc_freq[tid] += 1
        return tid

    def term_id(self, term: str) -> int:
        """Retorna o ID de um termo já registrado.

        Args:
            term: Termo a consultar.

        Returns:
            ID inteiro.

        Raises:
            KeyError: Se o termo não estiver no vocabulário.
        """
        return self._term_to_id[term]

    def id_term(self, tid: int) -> str:
        """Retorna o termo correspondente a um ID.

        Args:
            tid: ID do termo.

        Returns:
            String do termo.

        Raises:
            KeyError: Se o ID não existir.
        """
        return self._id_to_term[tid]

    def document_frequency(self, term: str) -> int:
        """Retorna a frequência de documento de um termo.

        Args:
            term: Termo consultado.

        Returns:
            Número de janelas em que o termo apareceu; 0 se não registrado.
        """
        tid = self._term_to_id.get(term)
        if tid is None:
            return 0
        return self._doc_freq[tid]

    def prune(self, min_df: int = 1, max_df_ratio: float = 1.0, n_windows: int = 0) -> "Vocabulary":
        """Remove termos fora dos limites de frequência e retorna novo Vocabulary.

        Args:
            min_df: Frequência mínima de documento (inclusivo).
            max_df_ratio: Frequência máxima como fração do total de janelas (0–1].
            n_windows: Total de janelas processadas. Se 0, usa o máximo observado
                entre as frequências como aproximação conservadora.

        Returns:
            Novo Vocabulary com apenas os termos dentro dos limites.
        """
        n = n_windows if n_windows > 0 else max(self._doc_freq.values(), default=1)
        max_abs = int(max_df_ratio * n)
        new_vocab = Vocabulary()
        for term, tid in self._term_to_id.items():
            freq = self._doc_freq[tid]
            if min_df <= freq <= max_abs:
                new_tid = new_vocab.add(term)
                new_vocab._doc_freq[new_tid] = freq
        return new_vocab

    def __len__(self) -> int:
        return self._next_id

    def __contains__(self, term: str) -> bool:
        return term in self._term_to_id

    def __iter__(self) -> Iterator[str]:
        return iter(self._term_to_id)

    def terms(self) -> list[str]:
        """Retorna todos os termos do vocabulário.

        Returns:
            Lista de strings.
        """
        return list(self._term_to_id.keys())
