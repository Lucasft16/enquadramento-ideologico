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


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lematiza uma lista de tokens usando spaCy.

    Processa os tokens como um único "documento" para aproveitar o pipeline
    vetorizado do spaCy. Tokens numéricos ou com um único caractere são
    descartados após a lematização.

    Args:
        tokens: Lista de tokens (strings) a lematizar.

    Returns:
        Lista de lemas, filtrando tokens inválidos.
    """
    if not tokens:
        return []
    nlp = _get_nlp()
    # Alimenta o modelo com os tokens já pré-tokenizados.
    doc = nlp.make_doc(" ".join(tokens))
    # Processa apenas o componente de lematização (morphologizer/tagger).
    for _, proc in nlp.pipeline:
        doc = proc(doc)

    result: list[str] = []
    for token in doc:
        lemma = token.lemma_.lower().strip()
        if lemma and not token.is_punct and not token.like_num and len(lemma) > 1:
            result.append(lemma)
    return result


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
