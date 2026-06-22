"""Interface web do classificador ideológico — execute com: streamlit run app.py"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

import streamlit as st

from src.config import load_config
from src.datastructures.trie import Trie
from src.model.reference_model import load as load_model
from src.parser.pipeline import process_document
from src.scoring.classifier import classify
from src.viz.render_interactive import _IDEOLOGY_COLORS, get_doc_graph_html

_MODEL_PATH = ROOT / "outputs" / "models" / "model.json"
_MARKERS_PATH = ROOT / "data" / "lexicons" / "markers.txt"
_CONFIG_PATH = ROOT / "config.yaml"


@st.cache_resource
def _load_resources():
    """Carrega modelo, config e trie uma única vez por sessão do servidor."""
    cfg = load_config(_CONFIG_PATH)
    model = load_model(_MODEL_PATH)
    trie = Trie()
    if _MARKERS_PATH.exists():
        for line in _MARKERS_PATH.read_text(encoding="utf-8").splitlines():
            phrase = line.strip()
            if phrase:
                trie.insert(phrase)
    return cfg, model, trie


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Enquadramento Ideológico",
    page_icon="🔬",
    layout="wide",
)

st.title("🔬 Enquadramento Ideológico")
st.caption("Classifica textos políticos usando grafos de co-ocorrência.")

# Sidebar
with st.sidebar:
    st.header("Configurações")
    method = st.radio(
        "Método de classificação",
        options=["graph", "jaccard"],
        format_func=lambda x: "Janela de contexto" if x == "graph" else "Jaccard",
        index=0,
        help=(
            "**Janela de contexto** — usa as janelas de co-ocorrência do documento para pontuar "
            "pares de termos ideológicos que aparecem juntos no mesmo contexto.\n\n"
            "**Jaccard** — usa apenas presença/ausência dos termos, sem considerar "
            "o contexto em que aparecem."
        ),
    )

    st.divider()

    if _MODEL_PATH.exists():
        try:
            _, model_info, _ = _load_resources()
            st.success("Modelo carregado")
            st.metric("Termos no vocabulário", model_info["vocab_size"])
            st.metric("Ideologias", len(model_info["ideology_terms"]))
            ideologies = list(model_info["ideology_terms"].keys())
            for ideo in ideologies:
                color = _IDEOLOGY_COLORS.get(ideo, _IDEOLOGY_COLORS["unknown"])
                st.markdown(
                    f'<span style="color:{color}">■</span> {ideo}',
                    unsafe_allow_html=True,
                )
        except Exception:
            st.error("Erro ao carregar modelo.")
    else:
        st.warning(
            "Modelo não encontrado.\n\nExecute:\n```\npython scripts/generate_corpus.py\n"
            "python scripts/build_model.py\n```"
        )

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------

text_input = st.text_area(
    "Cole ou digite o texto para análise:",
    height=220,
    placeholder=(
        "Exemplo: A privatização de estatais e a desregulação dos mercados "
        "promovem eficiência e competitividade econômica..."
    ),
)

col_btn, col_info = st.columns([1, 5])
with col_btn:
    run = st.button("Analisar", type="primary", use_container_width=True)
with col_info:
    if text_input.strip():
        words = len(text_input.split())
        st.caption(f"{words} palavras")

# ---------------------------------------------------------------------------
# Análise
# ---------------------------------------------------------------------------

if run:
    if not text_input.strip():
        st.warning("Insira um texto antes de analisar.")
        st.stop()

    if not _MODEL_PATH.exists():
        st.error("Modelo não encontrado. Veja as instruções na barra lateral.")
        st.stop()

    cfg, model, trie = _load_resources()

    with st.spinner("Processando texto..."):
        windows = process_document(
            text_input,
            trie=trie,
            window_size=cfg["window_size"],
            use_lemmatizer=False,
        )

    if not windows:
        st.warning(
            "Nenhuma janela de co-ocorrência gerada. "
            "O texto é muito curto ou composto apenas por stopwords."
        )
        st.stop()

    with st.spinner("Classificando..."):
        scores = classify(windows, model, method=method)

    dominant = max(scores, key=lambda k: scores[k])
    dominant_color = _IDEOLOGY_COLORS.get(dominant, _IDEOLOGY_COLORS["unknown"])

    # --- Resultado principal ---
    st.divider()
    st.markdown(
        f"### Enquadramento dominante: "
        f'<span style="color:{dominant_color}; font-weight:bold;">'
        f"{dominant.upper()} ({scores[dominant]*100:.1f}%)"
        f"</span>",
        unsafe_allow_html=True,
    )

    all_terms = list({tok for win in windows for tok in win})
    st.caption(f"Termos únicos no texto: {len(all_terms)} · Janelas geradas: {len(windows)} · Método: {method}")

    # --- Barras de pontuação ---
    st.subheader("Distribuição ideológica")
    for ideology, prob in sorted(scores.items(), key=lambda x: -x[1]):
        color = _IDEOLOGY_COLORS.get(ideology, _IDEOLOGY_COLORS["unknown"])
        col_label, col_bar, col_pct = st.columns([2, 6, 1])
        with col_label:
            st.markdown(
                f'<span style="color:{color}; font-weight:600;">■ {ideology}</span>',
                unsafe_allow_html=True,
            )
        with col_bar:
            st.progress(float(prob))
        with col_pct:
            st.markdown(f"**{prob*100:.1f}%**")

    # --- Grafo interativo (download) ---
    st.divider()
    st.subheader("Grafo de co-ocorrência")
    st.caption(
        "Nós coloridos por ideologia · Tamanho = centralidade no modelo · "
        "Espessura da aresta = co-ocorrências · Aresta tracejada = ponte entre ideologias"
    )
    with st.spinner("Gerando grafo..."):
        html = get_doc_graph_html(windows, model)
    st.download_button(
        label="Baixar grafo interativo (HTML)",
        data=html,
        file_name="grafo_coocorrencia.html",
        mime="text/html",
        use_container_width=True,
    )
    st.info("Abra o arquivo HTML baixado no navegador para visualizar o grafo interativo.")
