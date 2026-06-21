"""Classificação de documentos novos em ideologias usando o modelo de referência."""

from __future__ import annotations

from typing import Any

from src.scoring.doc_graph import build_doc_graph


def score_document_jaccard(
    terms: list[str],
    model: dict[str, Any],
) -> dict[str, float]:
    """Pontua um documento usando sobreposição ponderada estilo Jaccard.

    Para cada ideologia, calcula:
      score(ideo) = Σ centralidade(t)  para t ∈ (doc ∩ ideologia)
                   ─────────────────────────────────────────────────
                   Σ centralidade(t)  para t ∈ (doc ∪ ideologia)

    Considera apenas presença/ausência dos termos no documento — não usa
    a estrutura de co-ocorrência do texto novo.

    Args:
        terms: Lista de termos únicos do documento (após pipeline).
        model: Modelo de referência (saída de build_reference_model).

    Returns:
        Dicionário {ideologia: probabilidade} normalizado para soma 1.
    """
    ideology_terms: dict[str, dict[str, float]] = model["ideology_terms"]
    doc_set = set(terms)
    raw_scores: dict[str, float] = {}

    for ideology, term_scores in ideology_terms.items():
        ideo_set = set(term_scores.keys())
        intersection = doc_set & ideo_set
        union = doc_set | ideo_set

        num = sum(term_scores.get(t, 0.0) for t in intersection)
        denom = sum(term_scores.get(t, 0.0) for t in union if t in term_scores)
        raw_scores[ideology] = num / denom if denom > 0 else 0.0

    return _normalize(raw_scores)


def score_document_graph(
    windows: list[list[str]],
    model: dict[str, Any],
) -> dict[str, float]:
    """Pontua um documento usando o grafo de co-ocorrência das suas janelas.

    Constrói um grafo do próprio documento e, para cada ideologia, combina:
      - node_score: soma da centralidade de referência dos termos ideológicos
        presentes no documento.
      - edge_score: soma dos pesos das arestas do doc_graph entre pares de
        termos da mesma ideologia, ponderada pela centralidade de referência
        de cada ponta.

    Termos ideológicos que co-ocorrem juntos no documento (mesma janela)
    contribuem mais do que termos dispersos em parágrafos distantes.

    Args:
        windows: Janelas deslizantes do documento (saída de process_document).
        model: Modelo de referência.

    Returns:
        Dicionário {ideologia: probabilidade} normalizado para soma 1.
    """
    doc_graph = build_doc_graph(windows)
    ideology_terms: dict[str, dict[str, float]] = model["ideology_terms"]
    doc_vertices = set(doc_graph.vertices())
    raw_scores: dict[str, float] = {}

    for ideology, term_scores in ideology_terms.items():
        overlap = list(doc_vertices & set(term_scores.keys()))

        node_score = sum(term_scores[t] for t in overlap)

        edge_score = 0.0
        for i, u in enumerate(overlap):
            for j in range(i + 1, len(overlap)):
                v = overlap[j]
                w = doc_graph.edge_weight(u, v)
                if w > 0:
                    edge_score += w * term_scores[u] * term_scores[v]

        raw_scores[ideology] = node_score + edge_score

    return _normalize(raw_scores)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normaliza scores para soma 1; retorna distribuição uniforme se todos zero."""
    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {k: 1.0 / n if n > 0 else 0.0 for k in scores}
    return {k: v / total for k, v in scores.items()}


def classify(
    windows: list[list[str]],
    model: dict[str, Any],
    method: str = "graph",
) -> dict[str, float]:
    """Classifica um documento e retorna distribuição de ideologias.

    Args:
        windows: Janelas deslizantes do documento (saída de process_document).
        model: Modelo de referência.
        method: "graph" (padrão) usa co-ocorrência do documento;
                "jaccard" usa apenas presença/ausência de termos.

    Returns:
        Dicionário {ideologia: probabilidade} normalizado.

    Raises:
        ValueError: Se o método for desconhecido.
    """
    if method == "graph":
        return score_document_graph(windows, model)
    elif method == "jaccard":
        terms = list({tok for win in windows for tok in win})
        return score_document_jaccard(terms, model)
    else:
        raise ValueError(f"Método desconhecido: {method!r}")
