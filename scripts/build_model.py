"""FASE A — Constrói o modelo de referência ideológico a partir do corpus rotulado.

Execução:
    python scripts/build_model.py [--corpus CAMINHO] [--out CAMINHO]

Saída:
    outputs/models/model.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src.config import load_config
from src.datastructures.trie import Trie
from src.parser.pipeline import process_document
from src.model.reference_model import build_reference_model, save


def load_corpus(path: Path) -> list[dict]:
    """Carrega corpus JSONL."""
    docs = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


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
    parser = argparse.ArgumentParser(description="Constrói o modelo de referência ideológico.")
    parser.add_argument(
        "--corpus",
        default=str(ROOT / "data" / "raw" / "corpus.jsonl"),
        help="Caminho do corpus JSONL rotulado.",
    )
    parser.add_argument(
        "--seeds",
        default=str(ROOT / "data" / "lexicons" / "seeds.json"),
        help="Caminho do arquivo seeds.json.",
    )
    parser.add_argument(
        "--markers",
        default=str(ROOT / "data" / "lexicons" / "markers.txt"),
        help="Caminho dos marcadores multipalavra.",
    )
    parser.add_argument(
        "--out",
        default=str(ROOT / "outputs" / "models" / "model.json"),
        help="Caminho de saída do modelo JSON.",
    )
    parser.add_argument(
        "--config",
        default=str(ROOT / "config.yaml"),
        help="Caminho do config.yaml.",
    )
    parser.add_argument(
        "--label-weight",
        type=float,
        default=0.5,
        help=(
            "Peso dos rótulos do corpus vs. seeds na ancoragem supervisionada "
            "(0.0 = apenas seeds, 1.0 = apenas rótulos; padrão: 0.5)."
        ),
    )
    parser.add_argument(
        "--no-supervision",
        action="store_true",
        help="Ignora os rótulos do corpus e usa apenas as seeds (comportamento original).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    t0 = time.time()

    # 1. Carrega corpus
    corpus_path = Path(args.corpus)
    if not corpus_path.exists():
        print(f"[ERRO] Corpus não encontrado: {corpus_path}")
        print("       Execute: python scripts/generate_corpus.py")
        sys.exit(1)

    docs = load_corpus(corpus_path)
    print(f"Corpus carregado: {len(docs)} documentos")

    # 2. Extrai rótulos (campo "ideology") quando disponíveis
    doc_labels: list[str] | None = None
    if not args.no_supervision:
        labels = [doc.get("ideology") for doc in docs]
        if all(lbl is not None for lbl in labels):
            doc_labels = labels  # type: ignore[assignment]
            counts = Counter(doc_labels)
            print("Rótulos encontrados — treino supervisionado ativado:")
            for ideology, cnt in sorted(counts.items()):
                print(f"  {ideology}: {cnt} documentos")
            print(f"  label_weight={args.label_weight}")
        else:
            missing = sum(1 for lbl in labels if lbl is None)
            print(
                f"[AVISO] {missing}/{len(docs)} documentos sem campo 'ideology'. "
                "Usando apenas seeds (modo não-supervisionado)."
            )

    # Lematização DESLIGADA: o lematizador PT do spaCy é pouco confiável para o
    # vocabulário de domínio (gera lemas-lixo como "imposto"->"impor") e, no
    # corpus sintético, adiciona ruído à classificação. O código de lematização
    # (com proteção de marcadores e lematização consistente das sementes) está
    # pronto em src/parser/lemmatize.py para uso futuro com textos reais — basta
    # ligar aqui E em classify.py ao mesmo tempo (precisam casar).
    USE_LEMMATIZER = False

    # 3. Carrega seeds e Trie
    with open(args.seeds, "r", encoding="utf-8") as fh:
        seeds: dict[str, list[str]] = json.load(fh)
    if USE_LEMMATIZER:
        # As sementes precisam passar pela mesma lematização dos documentos,
        # senão deixam de casar com os vértices do grafo (ancoragem falharia).
        from src.parser.lemmatize import lemmatize_seeds

        seeds = lemmatize_seeds(seeds)
    trie = load_trie(Path(args.markers))
    print(f"Seeds: {list(seeds.keys())}")

    # 4. Processa documentos → janelas
    print(f"Processando documentos (lematizador={'spaCy' if USE_LEMMATIZER else 'simples'})...")
    docs_windows: list[list[list[str]]] = []
    for i, doc in enumerate(docs):
        windows = process_document(
            doc["text"],
            trie=trie,
            window_size=cfg["window_size"],
            use_lemmatizer=USE_LEMMATIZER,
        )
        docs_windows.append(windows)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{len(docs)} documentos processados")

    total_windows = sum(len(w) for w in docs_windows)
    print(f"Total de janelas: {total_windows}")

    # 5. Constrói modelo
    print(f"Construindo modelo (peso={cfg['weight_method']}, "
          f"filtro={cfg['filter_method']}, "
          f"comunidades={cfg['community_method']})...")

    model = build_reference_model(
        docs_windows=docs_windows,
        seeds=seeds,
        weight_method=cfg["weight_method"],
        filter_method=cfg["filter_method"],
        community_method=cfg["community_method"],
        threshold=cfg["threshold"],
        disparity_alpha=cfg.get("disparity_alpha", 0.05),
        max_communities=len(seeds) + 2,
        doc_labels=doc_labels,
        label_weight=args.label_weight,
        min_df=cfg.get("min_df", 1),
        max_df=cfg.get("max_df", 1.0),
    )

    # 6. Salva
    save(model, args.out)
    elapsed = time.time() - t0

    print(f"\nModelo salvo em: {args.out}")
    print(f"  Vocabulário: {model['vocab_size']} termos")
    print(f"  Arestas no grafo filtrado: {len(model['graph_edges'])}")
    print(f"  Comunidades detectadas: {len(model['communities'])}")
    print(f"  Ideologias: {list(model['ideology_terms'].keys())}")
    print(f"  Treino supervisionado: {'sim' if model.get('supervised') else 'não'}")
    print(f"  Tempo: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
