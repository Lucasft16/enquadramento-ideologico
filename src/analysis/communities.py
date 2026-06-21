"""Detecção de comunidades: Girvan-Newman e propagação de rótulos."""

from __future__ import annotations

import random
from collections import deque

from src.datastructures.graph import Graph
from src.analysis.traversal import connected_components
from src.analysis.centrality import betweenness_brandes


def modularity(graph: Graph, communities: list[list[str]]) -> float:
    """Calcula a modularidade de Newman-Girvan de uma partição.

    Q = Σ_{c} [ L_c/m - (d_c / 2m)^2 ]

    onde:
      m      = número total de arestas (contagem de pares únicos)
      L_c    = número de arestas internas à comunidade c
      d_c    = soma dos graus dos vértices de c (grau = número de vizinhos)

    Valores variam de -0.5 a 1.0; valores mais altos indicam partição melhor.
    Retorna 0.0 para grafos sem arestas ou com uma única comunidade trivial.

    Args:
        graph: Grafo não-direcionado ponderado.
        communities: Lista de comunidades; cada comunidade é lista de vértices.

    Returns:
        Modularidade Q ∈ [-0.5, 1.0].
    """
    m = graph.num_edges()
    if m == 0 or not communities:
        return 0.0

    two_m = 2.0 * m
    q = 0.0

    for community in communities:
        community_set = set(community)
        # Arestas internas: conta pares (u,v) em que ambos estão na comunidade.
        internal = sum(
            1
            for u in community
            for v in graph.neighbors(u)
            if v in community_set and u < v  # cada aresta contada uma vez
        )
        # Grau total dos vértices da comunidade (número de vizinhos, sem pesos).
        d_c = sum(graph.degree(v) for v in community)
        q += (internal / m) - (d_c / two_m) ** 2

    return q


def _edge_betweenness(graph: Graph) -> dict[tuple[str, str], float]:
    """Calcula a intermediação de arestas via adaptação de Brandes.

    Para cada vértice s, executa BFS e distribui crédito às arestas
    nos caminhos mínimos encontrados.

    Args:
        graph: Grafo não-direcionado.

    Returns:
        Dicionário {(u, v): intermediação} com u <= v lexicograficamente.
    """
    vertices = graph.vertices()
    eb: dict[tuple[str, str], float] = {}
    for e in graph.edges():
        key = (e[0], e[1])
        eb[key] = 0.0

    for s in vertices:
        stack: list[str] = []
        pred: dict[str, list[str]] = {v: [] for v in vertices}
        sigma: dict[str, float] = {v: 0.0 for v in vertices}
        sigma[s] = 1.0
        dist: dict[str, int] = {v: -1 for v in vertices}
        dist[s] = 0
        queue: deque[str] = deque([s])

        while queue:
            v = queue.popleft()
            stack.append(v)
            for w in graph.neighbors(v):
                if dist[w] < 0:
                    queue.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)

        delta: dict[str, float] = {v: 0.0 for v in vertices}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                contribution = (sigma[v] / sigma[w]) * (1.0 + delta[w])
                delta[v] += contribution
                # Distribui crédito à aresta (v, w).
                a, b = (v, w) if v <= w else (w, v)
                if (a, b) in eb:
                    eb[(a, b)] += contribution
            # delta[w] já foi acumulado nas iterações anteriores.

    # Normaliza (cada aresta foi contada duas vezes em grafos não-dir.)
    return {k: v / 2.0 for k, v in eb.items()}


def detect_communities(
    graph: Graph,
    max_communities: int = 10,
) -> list[list[str]]:
    """Girvan-Newman com parada automática por modularidade.

    Remove iterativamente a aresta de maior intermediação. A cada iteração,
    calcula a modularidade da partição atual. Para automaticamente quando a
    modularidade começa a cair (critério de hill-climbing), ou quando o grafo
    atinge `max_communities` componentes (teto de segurança), ou quando não
    há mais arestas.

    Essa abordagem elimina a necessidade de ajustar `max_communities` à mão:
    o algoritmo encontra sozinho a melhor partição segundo a modularidade de
    Newman-Girvan.

    Args:
        graph: Grafo ponderado não-direcionado (cópia interna é usada).
        max_communities: Teto máximo de comunidades (segurança); o critério
            principal de parada é a modularidade.

    Returns:
        Lista de comunidades; cada comunidade é uma lista de vértices.
    """
    g = graph.copy()

    best_partition: list[list[str]] = connected_components(g)
    best_q: float = modularity(g, best_partition)

    while True:
        comps = connected_components(g)
        if len(comps) >= max_communities:
            break
        if g.num_edges() == 0:
            break

        eb = _edge_betweenness(g)
        if not eb:
            break

        # Remove a aresta com maior intermediação.
        best_edge = max(eb, key=lambda e: eb[e])
        g.remove_edge(best_edge[0], best_edge[1])

        # Avalia a modularidade da nova partição.
        current_comps = connected_components(g)
        current_q = modularity(g, current_comps)

        if current_q > best_q:
            # Melhora: atualiza o melhor resultado.
            best_q = current_q
            best_partition = current_comps
        else:
            # Modularidade não melhorou: mantém a melhor partição encontrada.
            # Continua removendo arestas apenas se ainda não atingimos o teto.
            # (Permite passar de platôs temporários, mas para se piorar.)
            if len(current_comps) > len(best_partition):
                break

    return best_partition


def label_propagation(
    graph: Graph,
    max_iter: int = 100,
    seed: int = 42,
) -> list[list[str]]:
    """Detecção de comunidades por propagação de rótulos (assíncrona).

    Cada vértice adota o rótulo mais frequente entre seus vizinhos.
    Convergência quando nenhum rótulo muda ou após `max_iter` iterações.

    Args:
        graph: Grafo ponderado não-direcionado.
        max_iter: Número máximo de iterações.
        seed: Semente para reprodutibilidade.

    Returns:
        Lista de comunidades detectadas.
    """
    rng = random.Random(seed)
    vertices = graph.vertices()
    if not vertices:
        return []

    # Inicializa cada vértice com seu próprio rótulo.
    labels: dict[str, str] = {v: v for v in vertices}

    for _ in range(max_iter):
        order = list(vertices)
        rng.shuffle(order)
        changed = False

        for v in order:
            nbrs = graph.neighbors(v)
            if not nbrs:
                continue

            # Conta rótulos dos vizinhos ponderados pelo peso da aresta.
            label_scores: dict[str, float] = {}
            for u, w in nbrs.items():
                lbl = labels[u]
                label_scores[lbl] = label_scores.get(lbl, 0.0) + w

            best = max(label_scores, key=lambda l: (label_scores[l], l))
            if best != labels[v]:
                labels[v] = best
                changed = True

        if not changed:
            break

    # Agrupa vértices pelo rótulo final.
    groups: dict[str, list[str]] = {}
    for v, lbl in labels.items():
        groups.setdefault(lbl, []).append(v)

    result = list(groups.values())
    result.sort(key=len, reverse=True)
    return result
