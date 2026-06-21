"""Pipeline completo de processamento de texto: sanitize → stopwords → trie → lemmatize → windows."""

from __future__ import annotations

from typing import Optional

from src.config import CONFIG
from src.datastructures.trie import Trie
from src.parser.sanitize import sanitize
from src.parser.stopwords import remove_stopwords
from src.parser.windows import sliding_windows


def _apply_trie(tokens: list[str], trie: Optional[Trie]) -> list[str]:
    """Substitui sequências de tokens por marcadores multipalavra da Trie.

    Percorre os tokens da esquerda para a direita. Quando um marcador é
    encontrado, ele é inserido como um único token (espaços substituídos
    por underscore para manter compatibilidade com vocabulário).

    Args:
        tokens: Lista de tokens após remoção de stopwords.
        trie: Instância da Trie com marcadores; se None, retorna tokens sem alteração.

    Returns:
        Lista de tokens com marcadores multipalavra colapsados.
    """
    if trie is None:
        return tokens
    result: list[str] = []
    i = 0
    while i < len(tokens):
        phrase, nxt = trie.match_longest(tokens, i)
        if phrase is not None:
            # Colapsa espaços em underscore para formar token único.
            result.append(phrase.replace(" ", "_"))
            i = nxt
        else:
            result.append(tokens[i])
            i += 1
    return result


def process_document(
    text: str,
    trie: Optional[Trie] = None,
    window_size: Optional[int] = None,
    use_lemmatizer: bool = True,
) -> list[list[str]]:
    """Processa um documento de texto e retorna janelas de coocorrência.

    Pipeline aplicado:
    1. sanitize — normalização e limpeza.
    2. tokenização por espaço.
    3. remove_stopwords — filtro de stopwords.
    4. trie — colapso de marcadores multipalavra (opcional).
    5. lemmatize — lematização com spaCy (ou identidade se desativado).
    6. sliding_windows — geração de janelas de contexto.

    Args:
        text: Texto bruto do documento.
        trie: Trie de marcadores multipalavra (opcional).
        window_size: Tamanho da janela; usa config.yaml se None.
        use_lemmatizer: Se True, aplica lematização via spaCy.

    Returns:
        Lista de janelas, onde cada janela é uma lista de strings (termos).
    """
    ws = window_size if window_size is not None else CONFIG["window_size"]

    # 1. Sanitização e tokenização
    clean = sanitize(text)
    tokens = clean.split()

    # 2. Remoção de stopwords
    tokens = remove_stopwords(tokens)

    # 3. Casamento de marcadores multipalavra via Trie
    tokens = _apply_trie(tokens, trie)

    # 4. Lematização
    if use_lemmatizer and tokens:
        from src.parser.lemmatize import lemmatize_tokens  # lazy import

        tokens = lemmatize_tokens(tokens)
    else:
        from src.parser.lemmatize import lemmatize_tokens_simple

        tokens = lemmatize_tokens_simple(tokens)

    # 5. Janelas deslizantes
    return sliding_windows(tokens, ws)


def process_corpus(
    documents: list[str],
    trie: Optional[Trie] = None,
    window_size: Optional[int] = None,
    use_lemmatizer: bool = True,
) -> list[list[list[str]]]:
    """Processa uma lista de documentos, retornando janelas por documento.

    Args:
        documents: Lista de textos brutos.
        trie: Trie de marcadores (opcional).
        window_size: Tamanho da janela.
        use_lemmatizer: Se True, usa spaCy para lematização.

    Returns:
        Lista em que cada elemento é a saída de process_document para um documento.
    """
    return [
        process_document(doc, trie=trie, window_size=window_size, use_lemmatizer=use_lemmatizer)
        for doc in documents
    ]
