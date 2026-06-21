"""Geração de janelas deslizantes de contexto sobre uma lista de tokens."""

from __future__ import annotations


def sliding_windows(tokens: list[str], window_size: int) -> list[list[str]]:
    """Gera janelas deslizantes de tamanho fixo sobre uma lista de tokens.

    Cada janela contém no máximo `window_size` tokens. Janelas menores
    que 2 tokens são descartadas (sem coocorrências úteis).

    Args:
        tokens: Lista de tokens do documento.
        window_size: Número de tokens por janela.

    Returns:
        Lista de janelas, cada uma sendo uma lista de tokens.

    Exemplo:
        >>> sliding_windows(["a", "b", "c", "d"], 3)
        [['a', 'b', 'c'], ['b', 'c', 'd']]
    """
    if window_size < 2:
        raise ValueError("window_size deve ser >= 2")
    n = len(tokens)
    if n < 2:
        return []
    return [tokens[i : i + window_size] for i in range(n - window_size + 1)]


def sentence_windows(
    sentences: list[list[str]], window_size: int
) -> list[list[str]]:
    """Aplica janelas deslizantes a cada sentença separadamente.

    Garante que janelas não cruzem fronteiras de sentença.

    Args:
        sentences: Lista de sentenças, cada uma como lista de tokens.
        window_size: Tamanho da janela.

    Returns:
        Lista plana de todas as janelas geradas.
    """
    result: list[list[str]] = []
    for sent in sentences:
        if len(sent) < 2:
            continue
        if len(sent) <= window_size:
            # Frase mais curta que a janela: emite uma única janela com todos
            # os termos, em vez de descartá-la (sliding_windows retornaria []).
            result.append(sent)
        else:
            result.extend(sliding_windows(sent, window_size))
    return result
