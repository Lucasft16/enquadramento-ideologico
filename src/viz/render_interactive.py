"""Visualização interativa do grafo de co-ocorrência do documento via pyvis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.scoring.doc_graph import build_doc_graph

# Paleta por ideologia — alinhada com seeds.json e generate_corpus.py.
_IDEOLOGY_COLORS: dict[str, str] = {
    "libertarianismo":   "#377eb8",  # azul
    "conservadorismo":   "#e41a1c",  # vermelho
    "comunismo":         "#984ea3",  # roxo
    "social-democracia": "#4daf4a",  # verde
    "unknown":           "#aaaaaa",  # cinza
}

_PHYSICS_OPTIONS = {
    "physics": {
        "enabled": True,
        "barnesHut": {
            "gravitationalConstant": -3000,
            "centralGravity": 0.8,
            "springLength": 120,
            "springConstant": 0.05,
            "damping": 0.12,
        },
        "stabilization": {"iterations": 200, "fit": True},
    },
    "edges": {
        "smooth": {"type": "continuous"},
        "font": {"size": 10, "align": "middle", "strokeWidth": 2, "strokeColor": "#ffffff"},
        "scaling": {"min": 1, "max": 10},
    },
    "nodes": {
        "font": {"size": 13, "strokeWidth": 2, "strokeColor": "#ffffff"},
        "borderWidth": 2,
        "shadow": True,
    },
    "groups": {
        "libertarianismo":   {"color": {"background": "#377eb8", "border": "#1a5fa0", "highlight": {"background": "#5599cc", "border": "#1a5fa0"}}},
        "conservadorismo":   {"color": {"background": "#e41a1c", "border": "#a01010", "highlight": {"background": "#f05555", "border": "#a01010"}}},
        "comunismo":         {"color": {"background": "#984ea3", "border": "#6b2e7a", "highlight": {"background": "#b070bb", "border": "#6b2e7a"}}},
        "social-democracia": {"color": {"background": "#4daf4a", "border": "#2d7a2a", "highlight": {"background": "#70c86e", "border": "#2d7a2a"}}},
        "unknown":           {"color": {"background": "#dddddd", "border": "#aaaaaa", "highlight": {"background": "#eeeeee", "border": "#aaaaaa"}}},
    },
    "interaction": {
        "hover": True,
        "tooltipDelay": 100,
        "navigationButtons": True,
        "keyboard": True,
    },
}


def _build_term_ideology_map(model: dict[str, Any]) -> dict[str, tuple[str, float]]:
    """Retorna {termo: (ideologia, centralidade)} usando a ideology com maior score."""
    ideology_terms: dict[str, dict[str, float]] = model["ideology_terms"]
    result: dict[str, tuple[str, float]] = {}
    for ideology, terms in ideology_terms.items():
        for term, score in terms.items():
            existing = result.get(term)
            if existing is None or score > existing[1]:
                result[term] = (ideology, score)
    return result


def _normalize_sizes(
    term_map: dict[str, tuple[str, float]],
    min_size: float = 12,
    max_size: float = 40,
) -> dict[str, float]:
    """Normaliza centralidades para o intervalo [min_size, max_size]."""
    scores = {t: s for t, (_, s) in term_map.items()}
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {t: (min_size + max_size) / 2 for t in scores}
    return {
        t: min_size + (s - lo) / (hi - lo) * (max_size - min_size)
        for t, s in scores.items()
    }


def render_doc_graph_html(
    windows: list[list[str]],
    model: dict[str, Any],
    out_path: str | Path,
    min_edge_weight: int = 1,
) -> None:
    """Gera visualização interativa HTML do grafo de co-ocorrência do documento.

    Nós são os termos do documento. Cor representa a ideologia do termo no
    modelo de referência (cinza = sem correspondência no modelo). Tamanho
    representa a centralidade no modelo. Espessura das arestas representa
    quantas janelas do documento contêm aquele par de termos.

    Arestas entre termos de ideologias diferentes são desenhadas tracejadas —
    indicam pontes semânticas entre campos ideológicos.

    Args:
        windows: Janelas do documento (saída de process_document).
        model: Modelo de referência.
        out_path: Caminho do arquivo HTML de saída.
        min_edge_weight: Peso mínimo para exibir uma aresta (filtra ruído).
    """
    net = _build_network(windows, model, min_edge_weight)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out))


def _build_network(
    windows: list[list[str]],
    model: dict[str, Any],
    min_edge_weight: int = 1,
):
    """Constrói e retorna o objeto Network do pyvis (lógica compartilhada)."""
    try:
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError("pyvis não instalado. Execute: pip install pyvis") from e

    doc_graph = build_doc_graph(windows)
    term_ideology_map = _build_term_ideology_map(model)
    node_sizes = _normalize_sizes(term_ideology_map)

    net = Network(
        height="820px",
        width="100%",
        bgcolor="#f9f9f9",
        font_color="#222222",
        cdn_resources="in_line",
    )
    net.set_options(json.dumps(_PHYSICS_OPTIONS))

    for term in doc_graph.vertices():
        ideology, centrality = term_ideology_map.get(term, ("unknown", 0.0))
        size = node_sizes.get(term, 8)   # unknown fica menor (fallback 8 vs 12)
        shape = "ellipse" if ideology != "unknown" else "dot"
        tooltip = (
            f"<b>{term}</b><br>"
            f"Ideologia: <b>{ideology}</b><br>"
            f"Centralidade no modelo: {centrality:.4f}"
        )
        net.add_node(
            term,
            label=term,
            title=tooltip,
            size=size,
            shape=shape,
            group=ideology,
        )

    for u, v, w in doc_graph.edges():
        if w < min_edge_weight:
            continue
        ideo_u = term_ideology_map.get(u, ("unknown",))[0]
        ideo_v = term_ideology_map.get(v, ("unknown",))[0]
        is_bridge = ideo_u != ideo_v
        edge_color = _IDEOLOGY_COLORS.get(ideo_u, _IDEOLOGY_COLORS["unknown"])
        tooltip = (
            f"<b>{u}</b> — <b>{v}</b><br>"
            f"Co-ocorrências no documento: <b>{int(w)}</b>"
        )
        options: dict[str, Any] = {
            "value": w,
            "title": tooltip,
            # Só mostra o número sobre a aresta quando a co-ocorrência é repetida (w > 1).
            # Para w=1 (par apareceu em apenas 1 janela) o label seria "1" em toda aresta,
            # poluindo o grafo sem adicionar informação.
            "label": str(int(w)) if w > 1 else "",
        }
        if is_bridge:
            options["color"] = {"color": "#999999", "opacity": 0.5}
            options["dashes"] = True
        else:
            options["color"] = {"color": edge_color, "opacity": 0.7}

        net.add_edge(u, v, **options)

    return net


def get_doc_graph_html(
    windows: list[list[str]],
    model: dict[str, Any],
    min_edge_weight: int = 1,
) -> str:
    """Retorna o HTML do grafo interativo como string (sem salvar em disco).

    Útil para embuti-lo diretamente em interfaces como Streamlit via
    ``st.components.v1.html()``.

    Args:
        windows: Janelas do documento (saída de process_document).
        model: Modelo de referência.
        min_edge_weight: Peso mínimo para exibir uma aresta.

    Returns:
        String HTML auto-contida com JS/CSS embutidos (cdn_resources="in_line").
    """
    net = _build_network(windows, model, min_edge_weight)
    return net.generate_html()
