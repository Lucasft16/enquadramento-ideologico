"""Funções de ponderação de arestas: frequência bruta, NPMI e Jaccard."""

from __future__ import annotations

import math

from src.datastructures.graph import Graph
from src.graph_build.vocabulary import Vocabulary


# Tipo do dicionário de coocorrências: (id_a, id_b) → contagem
CoocDict = dict[tuple[int, int], int]


def frequency(
    cooc: CoocDict,
    vocab: Vocabulary,
    graph: Graph,
) -> Graph:
    """Pondera arestas pela frequência bruta de coocorrência.

    Args:
        cooc: Dicionário de coocorrências brutas.
        vocab: Vocabulário com mapeamento id→termo.
        graph: Grafo a preencher (modificado in-place e retornado).

    Returns:
        Grafo com pesos de frequência bruta.
    """
    for (a, b), cnt in cooc.items():
        u = vocab.id_term(a)
        v = vocab.id_term(b)
        graph.set_edge(u, v, float(cnt))
    return graph


def npmi(
    cooc: CoocDict,
    vocab: Vocabulary,
    graph: Graph,
    N: int,
) -> Graph:
    """Pondera arestas com NPMI (Normalized Pointwise Mutual Information).

    NPMI(a,b) = PMI(a,b) / -log P(a,b)
    PMI(a,b)  = log[ P(a,b) / (P(a) * P(b)) ]

    Valores variam de -1 (nunca coocorrem) a +1 (sempre juntos).
    Valores negativos são zerados (sem aresta). Termos com frequência zero
    são ignorados silenciosamente.

    Args:
        cooc: Dicionário de coocorrências brutas.
        vocab: Vocabulário com frequências de documento.
        graph: Grafo a preencher (modificado in-place e retornado).
        N: Número total de janelas processadas.

    Returns:
        Grafo com pesos NPMI ∈ [0, 1].
    """
    if N == 0:
        return graph

    for (a, b), cnt in cooc.items():
        u = vocab.id_term(a)
        v = vocab.id_term(b)
        df_a = vocab.document_frequency(u)
        df_b = vocab.document_frequency(v)

        if df_a == 0 or df_b == 0 or cnt == 0:
            continue

        p_ab = cnt / N
        p_a = df_a / N
        p_b = df_b / N

        pmi = math.log(p_ab / (p_a * p_b))
        denom = -math.log(p_ab)

        if denom <= 0:
            continue  # p_ab ≥ 1 (impossível em prática, guarda defensiva)

        score = pmi / denom  # ∈ [-1, 1]
        if score > 0:
            graph.set_edge(u, v, score)

    return graph


def jaccard(
    cooc: CoocDict,
    vocab: Vocabulary,
    graph: Graph,
) -> Graph:
    """Pondera arestas com o coeficiente de Jaccard.

    J(a,b) = |A ∩ B| / |A ∪ B|
           = cooc(a,b) / (df_a + df_b - cooc(a,b))

    Args:
        cooc: Dicionário de coocorrências brutas.
        vocab: Vocabulário com frequências de documento.
        graph: Grafo a preencher (modificado in-place e retornado).

    Returns:
        Grafo com pesos Jaccard ∈ (0, 1].
    """
    for (a, b), cnt in cooc.items():
        u = vocab.id_term(a)
        v = vocab.id_term(b)
        df_a = vocab.document_frequency(u)
        df_b = vocab.document_frequency(v)
        denom = df_a + df_b - cnt
        if denom <= 0:
            continue
        score = cnt / denom
        if score > 0:
            graph.set_edge(u, v, score)
    return graph


def build_weighted_graph(
    cooc: CoocDict,
    vocab: Vocabulary,
    method: str = "npmi",
    N: int = 0,
) -> Graph:
    """Constrói um grafo ponderado a partir das coocorrências.

    Args:
        cooc: Dicionário de coocorrências brutas.
        vocab: Vocabulário com frequências.
        method: Método de ponderação — "frequency", "npmi" ou "jaccard".
        N: Total de janelas (necessário para npmi).

    Returns:
        Grafo ponderado.

    Raises:
        ValueError: Se o método for desconhecido.
    """
    g = Graph()
    if method == "frequency":
        return frequency(cooc, vocab, g)
    elif method == "npmi":
        return npmi(cooc, vocab, g, N)
    elif method == "jaccard":
        return jaccard(cooc, vocab, g)
    else:
        raise ValueError(f"Método de ponderação desconhecido: {method!r}")
