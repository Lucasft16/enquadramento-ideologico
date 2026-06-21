"""Lematização de tokens usando spaCy (modelo pt_core_news_md)."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

# spaCy é importado de forma lazy para não forçar o carregamento do modelo
# quando o módulo é importado por testes que não precisam de NLP.
_nlp = None


def _get_nlp():
    """Carrega o modelo spaCy (singleton por processo)."""
    global _nlp
    if _nlp is None:
        import spacy  # noqa: PLC0415

        try:
            _nlp = spacy.load("pt_core_news_md", disable=["parser", "ner"])
        except OSError:
            # Fallback: modelo menor se o médio não estiver instalado.
            _nlp = spacy.load("pt_core_news_sm", disable=["parser", "ner"])
    return _nlp


@lru_cache(maxsize=8192)
def _lemma_one(token: str) -> str:
    """Lematiza um único token (com cache para reaproveitar termos repetidos).

    Lematizar token a token garante alinhamento posicional — essencial para
    reinserir, nas posições corretas, os tokens que não devem ser lematizados.
    """
    nlp = _get_nlp()
    doc = nlp(token)
    if len(doc) == 0:
        return token
    return doc[0].lemma_.lower().strip()


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lematiza uma lista de tokens usando spaCy.

    Tokens com underscore (marcadores multipalavra como ``livre_mercado`` e
    termos negados como ``nao_privatização``) são preservados sem alteração:
    o lematizador do spaCy os corromperia (ex.: ``livre_mercado`` ->
    ``livre_mercar``), quebrando o casamento com as sementes. Tokens numéricos
    ou de um único caractere são descartados.

    Args:
        tokens: Lista de tokens (strings) a lematizar.

    Returns:
        Lista de lemas, filtrando tokens inválidos e preservando marcadores.
    """
    if not tokens:
        return []
    result: list[str] = []
    for token in tokens:
        if "_" in token:
            result.append(token)  # marcador / negação — não lematizar
            continue
        lemma = _lemma_one(token)
        if lemma and not lemma.isdigit() and len(lemma) > 1:
            result.append(lemma)
    return result


def lemmatize_seeds(seeds: dict[str, list[str]]) -> dict[str, list[str]]:
    """Aplica a mesma lematização dos documentos às sementes de cada ideologia.

    Quando o corpus é lematizado, os vértices do grafo ficam em forma de lema
    (ex.: ``empresas`` -> ``empresa``, ``imposto`` -> ``impor``). As sementes
    precisam passar pela MESMA transformação, senão deixam de casar com os
    termos do grafo e a ancoragem falha. Marcadores com underscore são
    preservados (ver `lemmatize_tokens`).

    Args:
        seeds: Dicionário {ideologia: [sementes]}.

    Returns:
        Dicionário com as sementes lematizadas (sem duplicatas, ordem mantida).
    """
    out: dict[str, list[str]] = {}
    for ideology, terms in seeds.items():
        seen: set[str] = set()
        lemmas: list[str] = []
        for lemma in lemmatize_tokens(terms):
            if lemma not in seen:
                seen.add(lemma)
                lemmas.append(lemma)
        out[ideology] = lemmas
    return out


def lemmatize_tokens_simple(tokens: list[str]) -> list[str]:
    """Versão sem spaCy para testes ou ambientes sem o modelo instalado.

    Retorna os próprios tokens (sem lematização real), apenas filtrando
    tokens muito curtos ou numéricos.

    Args:
        tokens: Lista de tokens.

    Returns:
        Lista filtrada de tokens.
    """
    return [t for t in tokens if len(t) > 1 and not t.isdigit()]
