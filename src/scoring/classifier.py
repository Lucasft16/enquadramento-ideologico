"""Classificação de documentos novos em ideologias usando o modelo de referência."""

from __future__ import annotations

from typing import Any


def score_document_jaccard(
    terms: list[str],
    model: dict[str, Any],
) -> dict[str, float]:
    """Pontua um documento usando sobreposição ponderada estilo Jaccard.

    Para cada ideologia, calcula:
      score(ideo) = Σ centralidade(t)  para t ∈ (doc ∩ ideologia)
                   ─────────────────────────────────────────────────
                   Σ centralidade(t)  para t ∈ (doc ∪ ideologia)

    Os scores são normalizados para somar 1 (distribuição de probabilidade).

    Args:
        terms: Lista de termos do documento (após pipeline sem janelas).
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
        # Para termos do doc que não estão no modelo, score = 0.
        denom = sum(term_scores.get(t, 0.0) for t in union if t in term_scores)
        raw_scores[ideology] = num / denom if denom > 0 else 0.0

    return _normalize(raw_scores)


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    """Normaliza um dicionário de scores para que a soma seja 1.

    Args:
        scores: Scores brutos por ideologia.

    Returns:
        Dicionário normalizado; se todos zero, retorna distribuição uniforme.
    """
    total = sum(scores.values())
    if total <= 0:
        n = len(scores)
        return {k: 1.0 / n if n > 0 else 0.0 for k in scores}
    return {k: v / total for k, v in scores.items()}


def classify(
    terms: list[str],
    model: dict[str, Any],
    method: str = "jaccard",
) -> dict[str, float]:
    """Classifica um documento e retorna distribuição de ideologias.

    Args:
        terms: Termos do documento (pós-pipeline, sem janelas).
        model: Modelo de referência.
        method: "jaccard".

    Returns:
        Dicionário {ideologia: probabilidade} normalizado.

    Raises:
        ValueError: Se o método for desconhecido.
    """
    if method == "jaccard":
        return score_document_jaccard(terms, model)
    else:
        raise ValueError(f"Método desconhecido: {method!r}")
