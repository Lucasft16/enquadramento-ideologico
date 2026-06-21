"""Construção, serialização e desserialização do modelo de referência ideológico."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.datastructures.graph import Graph
from src.graph_build.vocabulary import Vocabulary
from src.graph_build.cooccurrence import count_cooccurrences, build_vocab_from_windows
from src.graph_build.weighting import build_weighted_graph
from src.analysis.filtering import max_spanning_backbone, threshold_filter, disparity_filter
from src.analysis.communities import detect_communities, label_propagation
from src.analysis.centrality import degree_centrality, betweenness_brandes
from src.model.anchoring import anchor_communities, build_ideology_term_map


def build_reference_model(
    docs_windows: list[list[list[str]]],
    seeds: dict[str, list[str]],
    weight_method: str = "npmi",
    filter_method: str = "kruskal",
    community_method: str = "girvan_newman",
    threshold: float = 0.1,
    disparity_alpha: float = 0.05,
    max_communities: int = 10,
) -> dict[str, Any]:
    """Constrói o modelo de referência a partir das janelas de coocorrência do corpus.

    Pipeline:
    1. Vocabulário a partir das janelas.
    2. Contagem de coocorrências.
    3. Ponderação do grafo.
    4. Filtragem do grafo.
    5. Detecção de comunidades.
    6. Ancoragem por sementes.
    7. Centralidade por comunidade.

    Args:
        docs_windows: Janelas por documento (saída do pipeline parser por documento).
        seeds: Dicionário {ideologia: [termos-semente]}.
        weight_method: "frequency" | "npmi" | "jaccard".
        filter_method: "kruskal" | "threshold" | "disparity".
        community_method: "girvan_newman" | "label_propagation".
        threshold: Limiar para threshold_filter.
        disparity_alpha: Alpha para disparity_filter.
        max_communities: Limite máximo de comunidades (Girvan-Newman).

    Returns:
        Dicionário do modelo com campos:
          - "ideology_terms": {ideologia → {termo → score}}.
          - "graph_edges": lista de (u, v, w) do grafo filtrado.
          - "communities": lista de listas de termos.
          - "assignment": {str(índice) → ideologia}.
          - "vocab_size": tamanho do vocabulário.
    """
    # 1 & 2. Vocabulário e coocorrências
    vocab = build_vocab_from_windows(docs_windows)
    N = sum(len(windows) for windows in docs_windows)
    cooc = count_cooccurrences(docs_windows, vocab)

    # 3. Ponderação
    graph = build_weighted_graph(cooc, vocab, method=weight_method, N=N)

    # 4. Filtragem
    if filter_method == "kruskal":
        filtered = max_spanning_backbone(graph)
    elif filter_method == "threshold":
        filtered = threshold_filter(graph, threshold)
    elif filter_method == "disparity":
        filtered = disparity_filter(graph, disparity_alpha)
    else:
        filtered = graph

    # 5. Comunidades
    if community_method == "girvan_newman":
        communities = detect_communities(filtered, max_communities=max_communities)
    else:
        communities = label_propagation(filtered)

    # 6. Ancoragem
    assignment = anchor_communities(communities, seeds)

    # 7. Centralidade
    centrality = degree_centrality(filtered)

    # 8. Mapa ideologia → termos
    ideology_terms = build_ideology_term_map(communities, assignment, centrality)

    return {
        "ideology_terms": ideology_terms,
        "graph_edges": [(u, v, w) for u, v, w in filtered.edges()],
        "communities": communities,
        "assignment": {str(k): v for k, v in assignment.items()},
        "vocab_size": len(vocab),
    }


def save(model: dict[str, Any], path: str | Path) -> None:
    """Serializa o modelo para JSON.

    Args:
        model: Dicionário do modelo (saída de build_reference_model).
        path: Caminho do arquivo de saída (.json).
    """
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(model, fh, ensure_ascii=False, indent=2)


def load(path: str | Path) -> dict[str, Any]:
    """Carrega um modelo previamente serializado.

    Args:
        path: Caminho do arquivo JSON.

    Returns:
        Dicionário do modelo.

    Raises:
        FileNotFoundError: Se o arquivo não existir.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Modelo não encontrado: {p}")
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def model_to_graph(model: dict[str, Any]) -> Graph:
    """Reconstrói o grafo filtrado a partir do modelo serializado.

    Args:
        model: Dicionário do modelo.

    Returns:
        Instância de Graph com as arestas do modelo.
    """
    g = Graph()
    for u, v, w in model["graph_edges"]:
        g.set_edge(u, v, w)
    return g
