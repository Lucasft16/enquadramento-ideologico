"""Pipeline completo de processamento de texto: sanitize → stopwords → trie → lemmatize → windows."""

from __future__ import annotations

from typing import Optional

from src.config import CONFIG
from src.datastructures.trie import Trie
from src.parser.sanitize import sanitize, split_sentences
from src.parser.stopwords import get_stopwords, remove_stopwords
from src.parser.windows import sentence_windows

# Palavras de negação em português. Note que "não", "nem" e "sem" também são
# stopwords — por isso a negação precisa ser tratada ANTES de remove_stopwords,
# senão o negador é descartado e "não privatização" colapsa em "privatização".
_NEGATORS: frozenset[str] = frozenset(
    {"não", "nao", "nunca", "jamais", "nem", "sem", "tampouco"}
)
# Prefixo aplicado ao termo negado (sem acento, marcador estável no vocabulário).
_NEG_PREFIX = "nao_"
# Quantos tokens descartáveis (stopwords/curtos) podem separar o negador do alvo.
_NEG_SCOPE = 2


def _mark_negation(tokens: list[str]) -> list[str]:
    """Marca o alvo de cada negação, formando um bigrama negativo explícito.

    Detecta negadores (ex.: "não", "nunca") e prefixa o próximo token de
    conteúdo com ``nao_``, transformando ``["não", "privatização"]`` em
    ``["nao_privatização"]``. Assim a negação deixa de ser invisível: o termo
    negado vira um vértice distinto do termo afirmado no grafo de coocorrência.

    O negador é removido do fluxo (negadores como "nunca"/"jamais" não são
    stopwords e sobreviveriam de outra forma). Tokens descartáveis entre o
    negador e o alvo (stopwords ou de 1 caractere) são pulados — até
    ``_NEG_SCOPE`` deles — e mantidos para a filtragem posterior normal.

    Args:
        tokens: Lista de tokens logo após a tokenização (antes das stopwords).

    Returns:
        Lista de tokens com os alvos de negação prefixados e os negadores
        removidos.
    """
    stopwords = get_stopwords()
    n = len(tokens)
    drop: set[int] = set()    # índices de negadores a remover
    negate: set[int] = set()  # índices de alvos a prefixar
    i = 0
    while i < n:
        if tokens[i] in _NEGATORS:
            drop.add(i)
            j = i + 1
            steps = 0
            while j < n and steps < _NEG_SCOPE:
                cand = tokens[j]
                if cand in _NEGATORS:
                    drop.add(j)  # encadeia negadores sem consumir o escopo
                    j += 1
                    continue
                if cand in stopwords or len(cand) <= 1:
                    j += 1
                    steps += 1
                    continue
                negate.add(j)  # alvo de conteúdo encontrado
                break
        i += 1

    result: list[str] = []
    for idx, tok in enumerate(tokens):
        if idx in drop:
            continue
        result.append(_NEG_PREFIX + tok if idx in negate else tok)
    return result


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
    mark_negation: bool = True,
) -> list[list[str]]:
    """Processa um documento de texto e retorna janelas de coocorrência.

    O documento é dividido em sentenças e cada uma é processada isoladamente,
    de modo que as janelas de coocorrência NÃO cruzam fronteiras de sentença.

    Pipeline por sentença:
    1. sanitize — normalização e limpeza.
    2. tokenização por espaço.
    3. trie — colapso de marcadores multipalavra (antes das stopwords, para que
       expressões com stopword interna, como "imposto de renda", se formem).
    4. mark_negation — marca bigramas negativos ("não X" → "nao_X").
    5. remove_stopwords — filtro de stopwords.
    6. lemmatize — lematização com spaCy (ou identidade se desativado).
    Por fim, sentence_windows gera as janelas sem cruzar sentenças.

    Args:
        text: Texto bruto do documento.
        trie: Trie de marcadores multipalavra (opcional).
        window_size: Tamanho da janela; usa config.yaml se None.
        use_lemmatizer: Se True, aplica lematização via spaCy.
        mark_negation: Se True, marca termos negados antes das stopwords.

    Returns:
        Lista de janelas, onde cada janela é uma lista de strings (termos).
    """
    ws = window_size if window_size is not None else CONFIG["window_size"]

    if use_lemmatizer:
        from src.parser.lemmatize import lemmatize_tokens as _lemma  # lazy import
    else:
        from src.parser.lemmatize import lemmatize_tokens_simple as _lemma

    sentences_tokens: list[list[str]] = []
    for sentence in split_sentences(text):
        # 1-2. Sanitização e tokenização
        tokens = sanitize(sentence).split()
        if not tokens:
            continue
        # 3. Marcadores multipalavra (antes das stopwords)
        tokens = _apply_trie(tokens, trie)
        # 4. Marcação de negação
        if mark_negation:
            tokens = _mark_negation(tokens)
        # 5. Remoção de stopwords
        tokens = remove_stopwords(tokens)
        # 6. Lematização (preserva marcadores/negações com underscore)
        tokens = _lemma(tokens)
        if tokens:
            sentences_tokens.append(tokens)

    # Janelas que não cruzam fronteiras de sentença.
    return sentence_windows(sentences_tokens, ws)


def process_corpus(
    documents: list[str],
    trie: Optional[Trie] = None,
    window_size: Optional[int] = None,
    use_lemmatizer: bool = True,
    mark_negation: bool = True,
) -> list[list[list[str]]]:
    """Processa uma lista de documentos, retornando janelas por documento.

    Args:
        documents: Lista de textos brutos.
        trie: Trie de marcadores (opcional).
        window_size: Tamanho da janela.
        use_lemmatizer: Se True, usa spaCy para lematização.
        mark_negation: Se True, marca termos negados antes das stopwords.

    Returns:
        Lista em que cada elemento é a saída de process_document para um documento.
    """
    return [
        process_document(
            doc,
            trie=trie,
            window_size=window_size,
            use_lemmatizer=use_lemmatizer,
            mark_negation=mark_negation,
        )
        for doc in documents
    ]
