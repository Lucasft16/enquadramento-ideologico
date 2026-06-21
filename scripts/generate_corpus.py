"""Gera corpus sintético rotulado por ideologia a partir de templates e léxicos.

Execução:
    python scripts/generate_corpus.py

Saída:
    data/raw/corpus.jsonl — um documento JSON por linha com campos:
        {"ideology": "...", "text": "..."}
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

SEED = 42
DOCS_PER_IDEOLOGY = 40

# Templates por ideologia — cada {X} será preenchido com termos do léxico.
TEMPLATES: dict[str, list[str]] = {
    "neoliberal": [
        "O {A} é essencial para garantir {B} e estimular {C} na economia.",
        "Sem {A}, não há {B}. A {C} depende de {D} eficiente.",
        "A {A} do setor público promove {B} e atrai {C} estrangeiro.",
        "Políticas de {A} e {B} são fundamentais para o {C} econômico.",
        "O governo deve reduzir {A} para aumentar {B} e {C}.",
        "A {A} fiscal cria ambiente favorável ao {B} e à {C}.",
        "Investidores buscam {A} e {B} antes de alocar {C}.",
        "A {A} de empresas estatais traz {B} e melhora {C}.",
        "Com {A} adequada, o {B} cresce e a {C} aumenta.",
        "O {A} livre promove {B} e distribui {C} de forma eficiente.",
    ],
    "progressista": [
        "A {A} é um direito fundamental que o Estado deve garantir a todos.",
        "Sem {A} pública universal, a {B} aprofunda-se e afeta os mais {C}.",
        "Trabalhadores precisam de {A} para enfrentar a {B} e garantir {C}.",
        "O investimento em {A} e {B} reduz {C} e promove justiça social.",
        "A {A} social protege os mais {B} e reduz a {C} estrutural.",
        "Políticas de {A} e {B} são necessárias para incluir os {C}.",
        "O {A} público garante {B} de qualidade para todos os cidadãos.",
        "A {A} coletiva e o {B} são pilares de uma sociedade justa.",
        "Com {A} forte, os {B} têm mais {C} e dignidade.",
        "O Estado deve garantir {A}, {B} e {C} como direitos básicos.",
    ],
    "conservador": [
        "A {A} é o alicerce da sociedade e deve ser preservada.",
        "Sem {A} e {B}, a {C} social se fragmenta perigosamente.",
        "Os {A} tradicionais garantem {B} e {C} para as gerações futuras.",
        "A {A} pública depende do respeito à {B} e à {C}.",
        "Valores de {A} e {B} são essenciais para manter a {C}.",
        "A {A} nacional exige {B} e respeito à {C} cultural.",
        "O fortalecimento da {A} e da {B} é imperativo para a {C}.",
        "A {A} moral orienta a {B} e sustenta a {C} civilizatória.",
        "Sem {A} e {B}, a {C} se dissolve em caos e relativismo.",
        "A {A} da {B} é responsabilidade de cada cidadão comprometido com a {C}.",
    ],
    "ancap": [
        "Toda {A} é uma violação da {B} individual e deve ser abolida.",
        "O {A} estatal destrói a {B} e impede contratos {C}.",
        "Sem {A} coercitiva, os indivíduos estabeleceriam {B} {C} voluntariamente.",
        "A {A} privada é inviolável; qualquer {B} constitui agressão.",
        "O {A} é ilegítimo porque se baseia em {B} e {C} sobre o indivíduo.",
        "Mercados {A} resolvem {B} sem necessidade de {C} estatal.",
        "A {A} de serviços elimina o {B} e garante {C} real.",
        "Cada indivíduo é {A} e tem direito à {B} de suas escolhas.",
        "O {A} monopoliza {B} e elimina alternativas {C}.",
        "Sem {A}, a {B} floresce e os {C} são desnecessários.",
    ],
}

LEXICONS: dict[str, list[str]] = {
    "neoliberal": [
        "mercado", "privatização", "eficiência", "competitividade", "reforma",
        "fiscal", "ajuste", "austeridade", "liberalização", "desregulação",
        "capital", "investimento", "produtividade", "exportação", "crescimento",
        "livre_mercado", "abertura", "concorrência", "lucro", "inovação",
    ],
    "progressista": [
        "desigualdade", "redistribuição", "direitos", "saúde", "educação",
        "moradia", "trabalhador", "sindicato", "proteção", "social",
        "público", "gratuidade", "inclusão", "vulnerável", "pobre",
        "previdência", "solidariedade", "equidade", "acesso", "comunidade",
    ],
    "conservador": [
        "família", "tradição", "valores", "ordem", "segurança",
        "autoridade", "religião", "nação", "pátria", "identidade",
        "costumes", "moral", "disciplina", "responsabilidade", "hierarquia",
        "civilização", "cultura", "herança", "respeito", "estabilidade",
    ],
    "ancap": [
        "liberdade", "propriedade", "voluntário", "contrato", "anarquia",
        "monopólio", "força", "coerção", "livre_mercado", "estatismo",
        "imposto", "regulação", "intervenção", "autônomo", "agência",
        "agressão", "estado", "soberania", "autodefesa", "consenso",
    ],
}


def _fill_template(template: str, lexicon: list[str], rng: random.Random) -> str:
    """Preenche os marcadores {A}, {B}, {C}, {D} do template com termos do léxico."""
    slots = ["A", "B", "C", "D"]
    chosen = rng.sample(lexicon, min(len(slots), len(lexicon)))
    result = template
    for slot, term in zip(slots, chosen):
        result = result.replace(f"{{{slot}}}", term.replace("_", " "), 1)
    return result


def generate_corpus(
    docs_per_ideology: int = DOCS_PER_IDEOLOGY,
    seed: int = SEED,
) -> list[dict[str, str]]:
    """Gera o corpus sintético rotulado.

    Args:
        docs_per_ideology: Número de documentos por ideologia.
        seed: Semente para reprodutibilidade.

    Returns:
        Lista de dicionários {"ideology": ..., "text": ...}.
    """
    rng = random.Random(seed)
    corpus: list[dict[str, str]] = []

    for ideology in TEMPLATES:
        templates = TEMPLATES[ideology]
        lexicon = LEXICONS[ideology]
        for _ in range(docs_per_ideology):
            # Compõe um documento com 3-5 frases.
            n_sentences = rng.randint(3, 5)
            sentences = [
                _fill_template(rng.choice(templates), lexicon, rng)
                for _ in range(n_sentences)
            ]
            text = " ".join(sentences)
            corpus.append({"ideology": ideology, "text": text})

    rng.shuffle(corpus)
    return corpus


def main() -> None:
    out_path = ROOT / "data" / "raw" / "corpus.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    corpus = generate_corpus()
    with out_path.open("w", encoding="utf-8") as fh:
        for doc in corpus:
            fh.write(json.dumps(doc, ensure_ascii=False) + "\n")

    print(f"Corpus gerado: {len(corpus)} documentos -> {out_path}")
    counts = {}
    for doc in corpus:
        counts[doc["ideology"]] = counts.get(doc["ideology"], 0) + 1
    for ideo, cnt in sorted(counts.items()):
        print(f"  {ideo}: {cnt} documentos")


if __name__ == "__main__":
    main()
