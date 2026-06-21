"""Lista de stopwords em português e função de filtragem."""

from __future__ import annotations

# Conjunto base de stopwords para o português do Brasil.
# Cobre artigos, preposições, pronomes, conjunções e verbos auxiliares comuns.
_STOPWORDS: frozenset[str] = frozenset(
    {
        "a", "ao", "aos", "aquela", "aquelas", "aquele", "aqueles", "aquilo",
        "as", "até", "com", "como", "da", "das", "de", "dela", "delas",
        "dele", "deles", "depois", "do", "dos", "e", "ela", "elas", "ele",
        "eles", "em", "entre", "era", "eram", "essa", "essas", "esse",
        "esses", "esta", "estas", "este", "estes", "eu", "foi", "foram",
        "há", "isso", "isto", "já", "lhe", "lhes", "mais", "mas", "me",
        "mesmo", "meu", "minha", "muito", "na", "nas", "nem", "no", "nos",
        "nossa", "nossas", "nosso", "nossos", "não", "num", "numa", "o",
        "os", "ou", "para", "pela", "pelas", "pelo", "pelos", "por", "pois",
        "qual", "quando", "que", "quem", "se", "seja", "sem", "seu", "seus",
        "si", "sobre", "sua", "suas", "são", "também", "te", "tem", "tendo",
        "ter", "tinha", "tudo", "um", "uma", "umas", "uns", "você", "vós",
        "à", "às", "é", "será", "seria", "ser", "ser", "está", "estão",
        "estava", "estavam", "temos", "tenho", "têm", "tinha",
    }
)


def get_stopwords() -> frozenset[str]:
    """Retorna o conjunto de stopwords em português.

    Returns:
        FrozenSet de strings com as stopwords.
    """
    return _STOPWORDS


def remove_stopwords(tokens: list[str]) -> list[str]:
    """Remove stopwords de uma lista de tokens.

    Args:
        tokens: Lista de tokens (já em minúsculas).

    Returns:
        Lista filtrada, sem stopwords.
    """
    return [t for t in tokens if t not in _STOPWORDS and len(t) > 1]
