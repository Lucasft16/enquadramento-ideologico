import unicodedata
import spacy
from spacy.lang.pt.stop_words import STOP_WORDS as STOPWORDS_SPACY

# Normalização de strings

def remover_acentos(texto: str) -> str:
    """Remove acentos preservando a letra base ('nação' -> 'nacao')."""
    decomposto = unicodedata.normalize("NFKD", texto)
    return "".join(c for c in decomposto if not unicodedata.combining(c))


def normalizar(texto: str) -> str:
    """Case folding + remoção de acentos. Usada de forma IDÊNTICA em texto e sementes."""
    return remover_acentos(texto.lower())

# Trie de FRASES (tokens), não de caracteres.

class _NoTrie:
    __slots__ = ("filhos", "fim", "token")

    def __init__(self):
        self.filhos = {}      # token (str) -> _NoTrie
        self.fim = False      # marca fim de uma frase-semente
        self.token = None     # token agrupado final, ex.: "luta_de_classe"


class TrieFrases:
    def __init__(self):
        self.raiz = _NoTrie()

    def inserir(self, tokens: list[str]) -> None:
        no = self.raiz
        for t in tokens:
            no = no.filhos.setdefault(t, _NoTrie())
        no.fim = True
        no.token = "_".join(tokens)

    def casar_maior(self, tokens: list[str], inicio: int) -> tuple[str | None, int]:
        """A partir de tokens[inicio], retorna (token_agrupado, qtd_consumida).
        Se nada casar, retorna (None, 0)."""
        no = self.raiz
        melhor_token, melhor_len = None, 0
        j = inicio
        while j < len(tokens) and tokens[j] in no.filhos:
            no = no.filhos[tokens[j]]
            j += 1
            if no.fim:                  # achou uma frase completa; guarda e tenta estender
                melhor_token, melhor_len = no.token, j - inicio
        return melhor_token, melhor_len

# Stopwords (hash set). Deacentuadas para casar com os tokens normalizados.

def carregar_stopwords(caminho: str | None = None) -> set[str]:
    """Lê stopwords de um arquivo (uma por linha). Sem arquivo, usa a lista do spaCy
    como ponto de partida. Tudo normalizado para bater com os tokens."""
    if caminho is None:
        return {normalizar(p) for p in STOPWORDS_SPACY}
    with open(caminho, encoding="utf-8") as f:
        return {normalizar(linha.strip()) for linha in f if linha.strip()}


# Preprocessador

class Preprocessador:
    def __init__(self, sementes: list[str], stopwords: set[str],
                 modelo: str = "pt_core_news_sm"):
        # 'ner' desativado por velocidade; mantemos o que separa sentenças e lematiza.
        self.nlp = spacy.load(modelo, disable=["ner"])
        self.stopwords = stopwords
        self.trie = TrieFrases()
        self._construir_trie(sementes)

    # etapa 1: spaCy -> pares (superfície, lema) normalizados por sentença 
    def _pares_por_sentenca(self, doc) -> list[list[tuple[str, str]]]:
        sentencas = []
        for sent in doc.sents:
            pares = []
            for tok in sent:
                if tok.is_punct or tok.is_space or tok.like_num:
                    continue
                surf = normalizar(tok.text).strip()
                lema = normalizar(tok.lemma_).strip()
                if surf:
                    pares.append((surf, lema))
            if pares:
                sentencas.append(pares)
        return sentencas

    # monta a Trie com a SUPERFÍCIE das sementes (mesmo pipeline) 
    def _construir_trie(self, sementes: list[str]) -> None:
        for frase in sementes:
            doc = self.nlp(frase)
            surfs = [s for sent in self._pares_por_sentenca(doc) for (s, _) in sent]
            if surfs:
                self.trie.inserir(surfs)

    # etapas 2+3: agrupa frases (por superfície) e finaliza os tokens 
    def _finalizar_sentenca(self, pares: list[tuple[str, str]]) -> list[str]:
        superficies = [s for (s, _) in pares]
        saida, i, n = [], 0, len(pares)
        while i < n:
            agrupado, consumido = self.trie.casar_maior(superficies, i)
            if agrupado:                       # frase ideológica: token fixo, sem lematizar
                saida.append(agrupado)
                i += consumido
                continue
            surf, lema = pares[i]
            i += 1
            # token solto: usa o lema; se o lema for contração ("de o"), cai pra superfície
            token = lema if (lema and " " not in lema) else surf
            if " " in token or len(token) <= 1:
                continue
            if surf in self.stopwords or token in self.stopwords:
                continue
            saida.append(token)
        return saida

    def _processar_doc(self, doc) -> list[list[str]]:
        resultado = []
        for pares in self._pares_por_sentenca(doc):
            tokens = self._finalizar_sentenca(pares)
            if tokens:
                resultado.append(tokens)
        return resultado

    def processar(self, texto: str) -> list[list[str]]:
        """Processa um único texto."""
        return self._processar_doc(self.nlp(texto))

    def processar_varios(self, textos: list[str]) -> list[list[list[str]]]:
        """Processa muitos textos de forma eficiente (usa nlp.pipe)."""
        return [self._processar_doc(doc) for doc in self.nlp.pipe(textos)]
