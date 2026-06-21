"""Testes para a ancoragem de comunidades a ideologias (src/model/anchoring.py)."""

from src.model.anchoring import anchor_communities, build_ideology_term_map


def test_anchor_communities_by_seed_majority():
    communities = [["mercado", "lucro", "privatização"], ["estado", "coletivização"]]
    seeds = {"neoliberal": ["mercado", "privatização"], "comunismo": ["coletivização"]}
    assignment = anchor_communities(communities, seeds)
    assert assignment[0] == "neoliberal"
    assert assignment[1] == "comunismo"


def test_seed_priority_overrides_community_label():
    # "desregulação" é semente neoliberal, mas caiu numa comunidade rotulada
    # como social-democracia. A prioridade de semente deve devolvê-la ao
    # neoliberal e removê-la da social-democracia.
    communities = [["bem_estar", "desregulação", "pacto"]]
    assignment = {0: "social-democracia"}
    centrality = {"bem_estar": 0.5, "desregulação": 0.3, "pacto": 0.4}
    seeds = {"neoliberal": ["desregulação"], "social-democracia": ["bem_estar", "pacto"]}

    term_map = build_ideology_term_map(communities, assignment, centrality, seeds)

    assert "desregulação" in term_map["neoliberal"]
    assert "desregulação" not in term_map.get("social-democracia", {})
    # Termos não-semente permanecem na comunidade.
    assert "bem_estar" in term_map["social-democracia"]


def test_seed_priority_ignores_absent_seeds():
    # Semente que não está no grafo (ausente da centralidade) é ignorada.
    communities = [["mercado"]]
    assignment = {0: "neoliberal"}
    centrality = {"mercado": 1.0}
    seeds = {"neoliberal": ["mercado"], "comunismo": ["proletariado"]}

    term_map = build_ideology_term_map(communities, assignment, centrality, seeds)

    assert "mercado" in term_map["neoliberal"]
    assert "proletariado" not in term_map.get("comunismo", {})
