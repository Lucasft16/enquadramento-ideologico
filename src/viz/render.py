"""Visualização do subgrafo do documento colorido por ideologia."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")  # backend sem display para ambientes headless
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Paleta de cores para até 10 ideologias + unknown.
_PALETTE: list[str] = [
    "#e41a1c", "#377eb8", "#4daf4a", "#984ea3",
    "#ff7f00", "#a65628", "#f781bf", "#999999",
    "#8dd3c7", "#ffffb3",
]
_UNKNOWN_COLOR = "#cccccc"


def _spring_layout(
    vertices: list[str],
    edges: list[tuple[str, str, float]],
    seed: int = 42,
    iterations: int = 50,
) -> dict[str, tuple[float, float]]:
    """Layout spring (Fruchterman-Reingold) implementado à mão.

    Usa forças de repulsão entre todos os pares e forças de atração apenas
    nas arestas. Posições inicializadas com distribuição determinística via
    semente.

    Args:
        vertices: Lista de vértices.
        edges: Lista de (u, v, peso).
        seed: Semente para posições iniciais.
        iterations: Número de iterações do algoritmo.

    Returns:
        Dicionário {vértice: (x, y)}.
    """
    import random
    import math

    rng = random.Random(seed)
    pos: dict[str, list[float]] = {
        v: [rng.uniform(-1, 1), rng.uniform(-1, 1)] for v in vertices
    }
    n = len(vertices)
    if n == 0:
        return {}
    if n == 1:
        return {vertices[0]: (0.0, 0.0)}

    k = math.sqrt(1.0 / n)  # constante de equilíbrio
    area = 1.0

    for step in range(iterations):
        t = area / (step + 1)  # temperatura decai com o tempo

        disp: dict[str, list[float]] = {v: [0.0, 0.0] for v in vertices}

        # Forças de repulsão (todos os pares).
        vlist = vertices
        for i, u in enumerate(vlist):
            for j in range(i + 1, len(vlist)):
                v = vlist[j]
                dx = pos[u][0] - pos[v][0]
                dy = pos[u][1] - pos[v][1]
                dist = math.sqrt(dx * dx + dy * dy) or 0.01
                rep = k * k / dist
                disp[u][0] += dx / dist * rep
                disp[u][1] += dy / dist * rep
                disp[v][0] -= dx / dist * rep
                disp[v][1] -= dy / dist * rep

        # Forças de atração (arestas).
        for eu, ev, ew in edges:
            if eu not in pos or ev not in pos:
                continue
            dx = pos[eu][0] - pos[ev][0]
            dy = pos[eu][1] - pos[ev][1]
            dist = math.sqrt(dx * dx + dy * dy) or 0.01
            attr = dist * dist / k
            disp[eu][0] -= dx / dist * attr
            disp[eu][1] -= dy / dist * attr
            disp[ev][0] += dx / dist * attr
            disp[ev][1] += dy / dist * attr

        # Aplica deslocamento limitado pela temperatura.
        for v in vlist:
            d = math.sqrt(disp[v][0] ** 2 + disp[v][1] ** 2) or 0.01
            pos[v][0] += disp[v][0] / d * min(d, t)
            pos[v][1] += disp[v][1] / d * min(d, t)
            # Limita ao quadrado [-1, 1].
            pos[v][0] = max(-1.0, min(1.0, pos[v][0]))
            pos[v][1] = max(-1.0, min(1.0, pos[v][1]))

    return {v: (pos[v][0], pos[v][1]) for v in vertices}


def render_document_subgraph(
    doc_terms: list[str],
    model: dict[str, Any],
    out_path: str | Path,
    seed: int = 42,
    max_nodes: int = 60,
) -> None:
    """Desenha o subgrafo do documento colorido por ideologia e salva como imagem.

    Vértices são os termos do documento que aparecem no grafo de referência.
    Cores representam a ideologia da comunidade. Arestas que ligam vértices de
    comunidades diferentes (pontes/conceitos disputados) são desenhadas em preto
    tracejado.

    Args:
        doc_terms: Termos do documento (pós-pipeline).
        model: Modelo de referência.
        out_path: Caminho do arquivo de saída (PNG ou SVG).
        seed: Semente para layout.
        max_nodes: Número máximo de nós a exibir (por relevância/centralidade).
    """
    from src.datastructures.graph import Graph as _Graph

    ideology_terms: dict[str, dict[str, float]] = model["ideology_terms"]
    all_edges: list[tuple[str, str, float]] = [
        (u, v, w) for u, v, w in model["graph_edges"]
    ]

    # Mapeia cada termo à sua ideologia (o de maior centralidade).
    term_ideology: dict[str, str] = {}
    for ideology, terms in ideology_terms.items():
        for term, score in terms.items():
            existing = term_ideology.get(term)
            if existing is None or score > ideology_terms.get(existing, {}).get(term, -1):
                term_ideology[term] = ideology

    ideologies = sorted(set(term_ideology.values()))
    color_map: dict[str, str] = {}
    for i, ideo in enumerate(ideologies):
        color_map[ideo] = _PALETTE[i % len(_PALETTE)]
    color_map["unknown"] = _UNKNOWN_COLOR

    # Seleciona vértices: termos do doc presentes no grafo de referência.
    ref_vertices = {v for u, v, w in all_edges} | {u for u, v, w in all_edges}
    doc_set = set(doc_terms) & ref_vertices

    # Se for grande demais, limita pelos mais centrais no modelo.
    if len(doc_set) > max_nodes:
        all_scores = {}
        for terms in ideology_terms.values():
            all_scores.update(terms)
        doc_set = set(
            sorted(doc_set, key=lambda t: all_scores.get(t, 0.0), reverse=True)[:max_nodes]
        )

    # Arestas dentro do subconjunto.
    sub_edges = [(u, v, w) for u, v, w in all_edges if u in doc_set and v in doc_set]

    if not doc_set:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Nenhum termo do documento encontrado no modelo",
                ha="center", va="center", transform=ax.transAxes, fontsize=12)
        ax.axis("off")
        plt.tight_layout()
        plt.savefig(str(out_path), dpi=150, bbox_inches="tight")
        plt.close()
        return

    vertices = sorted(doc_set)
    pos = _spring_layout(vertices, sub_edges, seed=seed)

    fig, ax = plt.subplots(figsize=(12, 9))
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Subgrafo do Documento — Enquadramento Ideológico", fontsize=14, pad=12)

    # Determina ideologia de cada vértice do subgrafo.
    ideo_of: dict[str, str] = {v: term_ideology.get(v, "unknown") for v in vertices}

    # Desenha arestas.
    for u, v, w in sub_edges:
        x0, y0 = pos[u]
        x1, y1 = pos[v]
        is_bridge = ideo_of.get(u) != ideo_of.get(v)
        lw = max(0.3, min(3.0, w * 5))
        if is_bridge:
            ax.plot([x0, x1], [y0, y1], color="black", linewidth=lw,
                    linestyle="--", alpha=0.5, zorder=1)
        else:
            color = color_map.get(ideo_of.get(u, "unknown"), _UNKNOWN_COLOR)
            ax.plot([x0, x1], [y0, y1], color=color, linewidth=lw,
                    alpha=0.4, zorder=1)

    # Desenha vértices e rótulos.
    for v in vertices:
        x, y = pos[v]
        ideo = ideo_of.get(v, "unknown")
        c = color_map.get(ideo, _UNKNOWN_COLOR)
        ax.scatter(x, y, s=200, color=c, edgecolors="white", linewidths=0.8, zorder=3)
        ax.text(x, y + 0.04, v, ha="center", va="bottom", fontsize=7,
                zorder=4, bbox=dict(boxstyle="round,pad=0.1", fc="white", alpha=0.6, ec="none"))

    # Legenda.
    patches = [
        mpatches.Patch(color=color_map.get(ideo, _UNKNOWN_COLOR), label=ideo)
        for ideo in ideologies
    ]
    patches.append(
        mpatches.Patch(color=_UNKNOWN_COLOR, label="unknown")
    )
    ax.legend(handles=patches, loc="upper left", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()


def render_score_bar(
    scores: dict[str, float],
    out_path: str | Path,
    title: str = "Distribuição Ideológica",
) -> None:
    """Gera gráfico de barras com a distribuição ideológica de um documento.

    Args:
        scores: Dicionário {ideologia: probabilidade}.
        out_path: Caminho de saída da imagem.
        title: Título do gráfico.
    """
    ideologies = sorted(scores, key=lambda k: scores[k], reverse=True)
    values = [scores[k] * 100 for k in ideologies]

    palette = _PALETTE[: len(ideologies)]
    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.barh(ideologies, values, color=palette, edgecolor="white")
    ax.set_xlabel("Probabilidade (%)")
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, 100)

    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
            f"{val:.1f}%", va="center", ha="left", fontsize=9,
        )

    plt.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close()
