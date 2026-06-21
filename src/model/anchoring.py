"""Ancoragem de comunidades a ideologias via sementes (seeds)."""

from __future__ import annotations


def anchor_communities(
    communities: list[list[str]],
    seeds: dict[str, list[str]],
) -> dict[int, str]:
    """Atribui cada comunidade à ideologia com maior sobreposição de sementes.

    Para cada comunidade, conta quantas sementes de cada ideologia ela contém.
    A ideologia com mais sementes "ganha" a comunidade. Empates são resolvidos
    pelo nome da ideologia (ordem lexicográfica) para reprodutibilidade.
    Comunidades sem nenhuma semente ficam rotuladas como "unknown".

    Args:
        communities: Lista de comunidades; cada comunidade é lista de vértices (termos).
        seeds: Dicionário {nome_ideologia: [termos-semente]}.

    Returns:
        Dicionário {índice_da_comunidade: nome_da_ideologia}.
    """
    assignment: dict[int, str] = {}

    for idx, community in enumerate(communities):
        community_set = set(community)
        best_ideology = "unknown"
        best_count = 0

        for ideology, seed_terms in seeds.items():
            count = sum(1 for s in seed_terms if s in community_set)
            if count > best_count or (
                count == best_count and count > 0 and ideology < best_ideology
            ):
                best_count = count
                best_ideology = ideology

        assignment[idx] = best_ideology

    return assignment


def build_ideology_term_map(
    communities: list[list[str]],
    assignment: dict[int, str],
    centrality_scores: dict[str, float],
) -> dict[str, dict[str, float]]:
    """Constrói o mapa {ideologia → {termo: centralidade}} a partir das comunidades.

    Termos que aparecem em múltiplas comunidades recebem o maior score.

    Args:
        communities: Lista de comunidades (listas de termos).
        assignment: Mapa {índice_comunidade: ideologia} de anchor_communities.
        centrality_scores: Mapa {termo: pontuação} de centralidade.

    Returns:
        Dicionário aninhado {ideologia: {termo: pontuação}}.
    """
    ideology_map: dict[str, dict[str, float]] = {}

    for idx, community in enumerate(communities):
        ideology = assignment.get(idx, "unknown")
        if ideology not in ideology_map:
            ideology_map[ideology] = {}
        for term in community:
            score = centrality_scores.get(term, 0.0)
            existing = ideology_map[ideology].get(term, -1.0)
            if score > existing:
                ideology_map[ideology][term] = score

    return ideology_map
