"""Ancoragem de comunidades a ideologias via sementes (seeds) e rótulos supervisionados."""

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


def compute_label_votes(
    communities: list[list[str]],
    docs_windows: list[list[list[str]]],
    doc_labels: list[str],
) -> list[dict[str, float]]:
    """Calcula votos de ideologia para cada comunidade a partir dos rótulos do corpus.

    Para cada comunidade, conta quantas janelas de documentos de cada ideologia
    contêm pelo menos um termo da comunidade. O resultado é normalizado por
    comunidade (distribuição de probabilidade).

    Isso usa o campo ``ideology`` do ``corpus.jsonl`` como sinal supervisionado
    para reforçar a ancoragem baseada em seeds.

    Args:
        communities: Lista de comunidades; cada comunidade é lista de vértices.
        docs_windows: Janelas por documento (mesma ordem que ``doc_labels``).
        doc_labels: Lista de rótulos ideológicos (um por documento).

    Returns:
        Lista de dicionários {ideologia: peso_normalizado}, um por comunidade.
    """
    n_communities = len(communities)
    # votes[i] = {ideologia: contagem de janelas que tocam a comunidade i}
    votes: list[dict[str, float]] = [{} for _ in range(n_communities)]

    # Pré-computa set de termos por comunidade para busca O(1).
    community_sets: list[set[str]] = [set(c) for c in communities]

    for doc_windows, label in zip(docs_windows, doc_labels):
        for window in doc_windows:
            window_set = set(window)
            for idx, comm_set in enumerate(community_sets):
                if window_set & comm_set:  # janela tem ao menos um termo da comunidade
                    votes[idx][label] = votes[idx].get(label, 0.0) + 1.0

    # Normaliza cada vetor de votos para que a soma seja 1.
    normalized: list[dict[str, float]] = []
    for vote_dict in votes:
        total = sum(vote_dict.values())
        if total > 0:
            normalized.append({k: v / total for k, v in vote_dict.items()})
        else:
            normalized.append({})

    return normalized


def anchor_communities_supervised(
    communities: list[list[str]],
    seeds: dict[str, list[str]],
    label_votes: list[dict[str, float]],
    label_weight: float = 0.5,
) -> dict[int, str]:
    """Atribui ideologias às comunidades combinando seeds e rótulos supervisionados.

    Combina dois sinais:
      1. Votos das seeds   — proporção de sementes da ideologia na comunidade.
      2. Votos dos rótulos — proporção de janelas rotuladas que tocam a comunidade.

    O score final é: ``(1 - label_weight) * seed_score + label_weight * label_score``.

    Comunidades sem nenhum sinal (nenhuma seed, nenhum rótulo) ficam como "unknown".

    Args:
        communities: Lista de comunidades; cada comunidade é lista de vértices.
        seeds: Dicionário {nome_ideologia: [termos-semente]}.
        label_votes: Saída de ``compute_label_votes`` — votos normalizados por comunidade.
        label_weight: Peso dos rótulos vs. seeds (0 = apenas seeds, 1 = apenas rótulos).
            Padrão: 0.5 (contribuição igual).

    Returns:
        Dicionário {índice_da_comunidade: nome_da_ideologia}.
    """
    all_ideologies = list(seeds.keys())
    assignment: dict[int, str] = {}

    for idx, community in enumerate(communities):
        community_set = set(community)

        # --- Sinal 1: seeds ---
        total_seeds = sum(len(terms) for terms in seeds.values()) or 1
        seed_scores: dict[str, float] = {}
        for ideology, seed_terms in seeds.items():
            hits = sum(1 for s in seed_terms if s in community_set)
            seed_scores[ideology] = hits / total_seeds

        # Normaliza seed_scores para soma 1 (se houver algum hit).
        seed_total = sum(seed_scores.values())
        if seed_total > 0:
            seed_scores = {k: v / seed_total for k, v in seed_scores.items()}

        # --- Sinal 2: rótulos do corpus ---
        lv = label_votes[idx] if idx < len(label_votes) else {}

        # --- Combinação ponderada ---
        combined: dict[str, float] = {}
        for ideology in all_ideologies:
            s = seed_scores.get(ideology, 0.0)
            l = lv.get(ideology, 0.0)
            combined[ideology] = (1.0 - label_weight) * s + label_weight * l

        # Ideologia vencedora; empates resolvidos lexicograficamente.
        best_ideology = "unknown"
        best_score = 0.0
        for ideology, score in combined.items():
            if score > best_score or (
                score == best_score and score > 0 and ideology < best_ideology
            ):
                best_score = score
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
