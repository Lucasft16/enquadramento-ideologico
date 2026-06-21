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
    "libertarianismo": [
        "Toda {A} é uma violação da {B} individual e deve ser abolida.",
        "O {A} estatal destrói a {B} e impede contratos {C}.",
        "Sem {A} coercitiva, os indivíduos estabeleceriam {B} {C} voluntariamente.",
        "A {A} privada é inviolável; qualquer {B} constitui agressão.",
        "O livre {A} promove {B} e distribui {C} de forma eficiente.",
        "Mercados {A} resolvem {B} sem necessidade de {C} estatal.",
        "A {A} de empresas estatais aumenta {B} e {C}.",
        "Cada indivíduo é {A} e tem direito à {B} de suas escolhas.",
        "Com menos {A}, a {B} e a {C} do setor privado florescem.",
        "Sem {A}, a {B} prospera e os controles {C} são desnecessários.",
    ],
    "conservadorismo": [
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
    "comunismo": [
        "A {A} só será superada com a {B} promovida pela {C}.",
        "O {A} explora a {B} e concentra o poder sobre a {C}.",
        "A {A} dos meios de produção porá fim à {B} e à {C}.",
        "Sem a {A} do {B}, a {C} dominante permanecerá intacta.",
        "A {A} é o motor da história rumo ao {B} e ao fim da {C}.",
        "Camaradas organizam a {A} contra a {B} e a {C} de classe.",
        "Apenas a {A} coletiva elimina a {B} e a {C} do trabalhador.",
        "A {A} burguesa mascara a {B} e perpetua a {C}.",
        "A {A} do {B} é condição para uma sociedade sem {C}.",
        "Defendemos a {A} e a {B} contra o {C} e a propriedade privada.",
    ],
    "social-democracia": [
        "A {A} entre capital e trabalho garante {B} e {C}.",
        "O Estado promove {A} e {B} sem abolir o mercado.",
        "Por meio do {A} e da {B}, alcança-se a {C} social.",
        "A {A} regulada concilia {B} e {C} de forma gradual.",
        "Defendemos {A}, {B} e {C} pela via da negociação.",
        "O {A} entre os setores assegura {B} e reduz conflitos.",
        "A {A} democrática constrói {B} por meio do {C}.",
        "Com {A} e {B}, o progresso é {C} e sustentável.",
        "A {A} social resulta do {B} e da {C} entre as partes.",
        "Buscamos {A} e {B} respeitando a {C} institucional.",
    ],
}

LEXICONS: dict[str, list[str]] = {
    "conservadorismo": [
        "família", "tradição", "valores", "ordem", "segurança",
        "autoridade", "religião", "nação", "pátria", "identidade",
        "costumes", "moral", "disciplina", "responsabilidade", "hierarquia",
        "civilização", "cultura", "herança", "respeito", "estabilidade",
        "soberania", "segurança_pública",
    ],
    "libertarianismo": [
        "liberdade", "propriedade", "voluntário", "contrato", "mercado",
        "privatização", "eficiência", "competitividade", "desregulação", "autônomo",
        "anarquia", "estatismo", "individualismo", "empreendedorismo", "espontâneo",
        "capital", "lucro", "concorrência",
        "livre_mercado", "estado_mínimo", "propriedade_privada", "livre_comércio",
        "economia_de_mercado", "livre_iniciativa",
    ],
    "comunismo": [
        "proletariado", "burguesia", "revolução", "exploração", "coletivização",
        "estatização", "socialismo", "comunismo", "expropriação", "vanguarda",
        "classe", "opressão", "planificação", "camarada", "marxismo",
        "dialética", "alienação", "internacionalismo", "revolucionário", "ditadura",
        "economia_planificada", "propriedade_coletiva",
    ],
    "social-democracia": [
        "reformismo", "welfare", "seguridade", "redistribuição", "desigualdade",
        "direitos", "saúde", "educação", "sindicato", "conciliação",
        "gradualismo", "cidadania", "solidariedade", "proteção", "inclusão",
        "reformista", "justiça",
        "bem_estar_social", "previdência_social", "sistema_único_de_saúde",
        "direitos_humanos", "salário_mínimo",
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
