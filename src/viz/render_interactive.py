"""Visualização interativa do grafo de co-ocorrência do documento via pyvis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from src.scoring.doc_graph import build_doc_graph

# Paleta por ideologia — mesma ordem que render.py para consistência visual.
_IDEOLOGY_COLORS: dict[str, str] = {
    "neoliberal":   "#377eb8",  # azul
    "progressista": "#4daf4a",  # verde
    "conservador":  "#e41a1c",  # vermelho
    "ancap":        "#ff7f00",  # laranja
    "unknown":      "#aaaaaa",  # cinza
}

_PHYSICS_OPTIONS = {
    "physics": {
        "enabled": True,
        "barnesHut": {
            "gravitationalConstant": -6000,
            "centralGravity": 0.3,
            "springLength": 130,
            "springConstant": 0.04,
            "damping": 0.09,
        },
        "stabilization": {"iterations": 150},
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
    try:
        from pyvis.network import Network
    except ImportError as e:
        raise ImportError("pyvis não instalado. Execute: pip install pyvis") from e

    doc_graph = build_doc_graph(windows)
    term_ideology_map = _build_term_ideology_map(model)
    node_sizes = _normalize_sizes(term_ideology_map)

    net = Network(
        height="780px",
        width="100%",
        bgcolor="#f8f8f8",
        font_color="#222222",
        select_menu=True,
        filter_menu=True,
        neighborhood_highlight=True,
        cdn_resources="in_line",
    )
    net.set_options(json.dumps(_PHYSICS_OPTIONS))

    # Adiciona nós
    for term in doc_graph.vertices():
        ideology, centrality = term_ideology_map.get(term, ("unknown", 0.0))
        color = _IDEOLOGY_COLORS.get(ideology, _IDEOLOGY_COLORS["unknown"])
        size = node_sizes.get(term, 12)
        shape = "ellipse" if ideology != "unknown" else "diamond"
        tooltip = (
            f"<b>{term}</b><br>"
            f"Ideologia: <b>{ideology}</b><br>"
            f"Centralidade no modelo: {centrality:.4f}"
        )
        net.add_node(
            term,
            label=term,
            title=tooltip,
            color=color,
            size=size,
            shape=shape,
            group=ideology,
        )

    # Adiciona arestas
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
            "label": str(int(w)),
        }
        if is_bridge:
            options["color"] = {"color": "#999999", "opacity": 0.5}
            options["dashes"] = True
        else:
            options["color"] = {"color": edge_color, "opacity": 0.7}

        net.add_edge(u, v, **options)

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(out))
