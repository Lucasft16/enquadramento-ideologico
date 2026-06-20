"""FASE B — Classifica um documento novo e gera visualização do subgrafo colorido.

Execução:
    python scripts/classify.py data/examples/exemplo.txt [--model CAMINHO] [--method jaccard|dijkstra]

Saída:
    Distribuição ideológica impressa no terminal.
    outputs/figures/<nome_arquivo>.png — subgrafo colorido.
    outputs/figures/<nome_arquivo>_bars.png — gráfico de barras da distribuição.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.datastructures.trie import Trie
from src.parser.pipeline import process_document
from src.model.reference_model import load
from src.scoring.classifier import classify
from src.viz.render import render_document_subgraph, render_score_bar


def load_trie(markers_path: Path) -> Trie:
    """Carrega marcadores multipalavra na Trie."""
    trie = Trie()
    if markers_path.exists():
        with markers_path.open("r", encoding="utf-8") as fh:
            for line in fh:
                phrase = line.strip()
                if phrase:
                    trie.insert(phrase)
    return trie


def main() -> None:
    parser = argparse.ArgumentParser(description="Classifica documento e gera visualização.")
    parser.add_argument("document", help="Caminho do documento de texto a classificar.")
    parser.add_argument(
        "--model",
        default=str(ROOT / "outputs" / "models" / "model.json"),
        help="Caminho do modelo JSON.",
    )
    parser.add_argument(
        "--markers",
        default=str(ROOT / "data" / "lexicons" / "markers.txt"),
        help="Caminho dos marcadores multipalavra.",
    )
    parser.add_argument(
        "--method",
        choices=["jaccard", "dijkstra"],
        default="jaccard",
        help="Método de classificação.",
    )
    parser.add_argument(
        "--out-dir",
        default=str(ROOT / "outputs" / "figures"),
        help="Diretório de saída das figuras.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Caminho do config.yaml.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)

    # 1. Carrega modelo
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERRO] Modelo não encontrado: {model_path}")
        print("       Execute: python scripts/build_model.py")
        sys.exit(1)

    model = load(model_path)
    print(f"Modelo carregado: {model['vocab_size']} termos, "
          f"{len(model['graph_edges'])} arestas")

    # 2. Lê documento
    doc_path = Path(args.document)
    if not doc_path.exists():
        print(f"[ERRO] Documento não encontrado: {doc_path}")
        sys.exit(1)

    text = doc_path.read_text(encoding="utf-8")
    print(f"Documento: {doc_path.name} ({len(text)} caracteres)")

    # 3. Processa
    trie = load_trie(Path(args.markers))
    windows = process_document(
        text,
        trie=trie,
        window_size=cfg["window_size"],
        use_lemmatizer=False,
    )
    # Extrai todos os termos únicos do documento (sem janelas).
    all_terms: list[str] = list({tok for win in windows for tok in win})
    print(f"Termos unicos no documento: {len(all_terms)}")

    # 4. Classifica
    scores = classify(all_terms, model, method=args.method)

    # 5. Exibe resultado
    print("\n" + "=" * 50)
    print("DISTRIBUICAO IDEOLOGICA")
    print("=" * 50)
    for ideology, prob in sorted(scores.items(), key=lambda x: -x[1]):
        bar = "#" * int(prob * 40)
        print(f"  {ideology:20s} {prob * 100:5.1f}%  {bar}")
    print("=" * 50)

    dominant = max(scores, key=lambda k: scores[k])
    print(f"\nEnquadramento dominante: {dominant} ({scores[dominant]*100:.1f}%)")


    # 6. Visualizações
    out_dir = Path(args.out_dir)
    stem = doc_path.stem

    subgraph_path = out_dir / f"{stem}.png"
    render_document_subgraph(all_terms, model, subgraph_path, seed=cfg.get("seed", 42))
    print(f"\nSubgrafo salvo em: {subgraph_path}")

    bars_path = out_dir / f"{stem}_bars.png"
    render_score_bar(scores, bars_path, title=f"Distribuição — {doc_path.name}")
    print(f"Gráfico de barras salvo em: {bars_path}")


if __name__ == "__main__":
    main()
