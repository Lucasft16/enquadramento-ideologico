"""Contagem de coocorrências a partir de janelas de contexto."""

from __future__ import annotations

from src.graph_build.vocabulary import Vocabulary


def count_cooccurrences(
    docs_windows: list[list[list[str]]],
    vocab: Vocabulary,
) -> dict[tuple[int, int], int]:
    """Conta coocorrências de pares de termos em todas as janelas de todos os documentos.

    Para cada janela, todos os pares não-ordenados de termos distintos são
    contabilizados. O par é armazenado com o menor ID primeiro (ordenação canônica).

    Args:
        docs_windows: Lista de documentos; cada documento é uma lista de janelas;
                      cada janela é uma lista de strings (termos).
        vocab: Vocabulário que mapeia termos para IDs.

    Returns:
        Dicionário {(id_menor, id_maior): contagem}.
    """
    cooc: dict[tuple[int, int], int] = {}

    for doc_windows in docs_windows:
        for window in doc_windows:
            # Registra ocorrências no vocabulário e obtém IDs.
            ids: list[int] = []
            seen_in_window: set[int] = set()
            for term in window:
                if term in vocab:
                    tid = vocab.term_id(term)
                    ids.append(tid)
                    seen_in_window.add(tid)

            # Conta todos os pares dentro da janela.
            unique_ids = list(seen_in_window)
            n = len(unique_ids)
            for i in range(n):
                for j in range(i + 1, n):
                    a, b = unique_ids[i], unique_ids[j]
                    key = (a, b) if a < b else (b, a)
                    cooc[key] = cooc.get(key, 0) + 1

    return cooc


def build_vocab_from_windows(
    docs_windows: list[list[list[str]]],
) -> Vocabulary:
    """Constrói um Vocabulary registrando ocorrências a partir de janelas.

    Args:
        docs_windows: Janelas por documento.

    Returns:
        Vocabulary preenchido com frequências de documento.
    """
    vocab = Vocabulary()
    for doc_windows in docs_windows:
        for window in doc_windows:
            seen = set(window)
            for term in seen:
                vocab.register_occurrence(term)
    return vocab
