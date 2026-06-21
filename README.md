# Ideologia Grafos

Sistema que lê um texto qualquer e estima, em porcentagem, qual é o seu enquadramento ideológico — **neoliberal**, **progressista**, **conservador** ou **ancap** — usando grafos de coocorrência de palavras.

Desenvolvido como trabalho da disciplina de **Estruturas de Dados 2**. Todos os algoritmos e estruturas foram implementados do zero: grafo com lista de adjacência, Union-Find, Trie, BFS/DFS, Kruskal, Brandes, Dijkstra e Girvan-Newman.

---

## Como o sistema funciona

O sistema opera em duas fases:

**Fase A — Construção do modelo de referência (feita uma vez)**

Um corpus de textos é processado para montar um grafo onde os nós são palavras e as arestas representam a força com que duas palavras coocorrem nas mesmas frases. Esse grafo é filtrado, dividido em comunidades de palavras, e cada comunidade é associada a uma ideologia usando palavras-âncora definidas em `data/lexicons/seeds.json`. O resultado é salvo como `outputs/models/model.json`.

**Fase B — Classificação de um novo texto (feita para cada texto)**

O texto de entrada passa pelo mesmo pipeline de limpeza e tem seus termos comparados contra o modelo. O sistema retorna uma distribuição de probabilidade entre as ideologias e gera imagens do subgrafo.

---

## Estrutura do projeto

```
config.yaml               parâmetros globais (janela, método de peso, filtro)
data/
  lexicons/
    seeds.json            palavras-âncora de cada ideologia
    markers.txt           expressões multipalavra (ex: "livre mercado")
  examples/               textos de exemplo para classificar
  raw/                    corpus gerado automaticamente pelo script
src/
  config.py               carrega o config.yaml
  datastructures/         Graph, UnionFind, Trie
  parser/                 limpeza, stopwords, lematização, janelas deslizantes
  graph_build/            vocabulário, contagem de coocorrências, ponderação
  analysis/               BFS/DFS, Kruskal, Brandes, Dijkstra, Girvan-Newman
  model/                  ancoragem de comunidades e serialização do modelo
  scoring/                classificação por co-ocorrência de grafo (padrão) ou Jaccard
  viz/                    geração de imagens
scripts/
  generate_corpus.py      gera corpus sintético de treino
  build_model.py          executa a Fase A e salva o modelo
  classify.py             executa a Fase B e exibe o resultado
outputs/                  modelos e figuras gerados (criado automaticamente)
tests/                    testes unitários
```

---

## Como foi construído — sequência de commits

A ordem de commits reflete a ordem de dependências do sistema, algoritmo por algoritmo:

| # | Commit | O que entrou |
|---|--------|--------------|
| 1 | `feat: configuração inicial do projeto` | pyproject.toml, config.yaml, src/config.py |
| 2 | `feat: grafo ponderado não-direcionado com lista de adjacência` | `Graph` — estrutura central de tudo |
| 3 | `feat: UnionFind e Trie` | Union-Find para Kruskal; Trie para expressões multipalavra |
| 4 | `feat: vocabulário, coocorrências e ponderação de arestas` | Vocabulary, contagem de pares, NPMI/Jaccard |
| 5 | `feat: BFS, DFS e componentes conexos` | Travessias do grafo |
| 6 | `feat: filtragem por Kruskal, threshold e disparity` | Remoção de arestas fracas |
| 7 | `feat: centralidade de grau e Brandes` | Importância de cada nó no grafo |
| 8 | `feat: Dijkstra com custo inverso ao peso` | Menor caminho semântico |
| 9 | `feat: Girvan-Newman e propagação de rótulos` | Detecção de comunidades |
| 10 | `feat: modelo de referência e ancoragem` | Tudo se conecta; salva model.json |
| 11 | `feat: classificador (Jaccard e Dijkstra)` | Pontua um texto contra o modelo |
| 12 | `feat: visualização do subgrafo e barras` | Geração de imagens PNG |
| 13 | `chore: léxicos e exemplo` | seeds.json, markers.txt, exemplo.txt |
| 14 | `feat: scripts de execução` | generate_corpus, build_model, classify |
| 15 | `feat: pipeline de pré-processamento de texto` | sanitize, stopwords, janelas, pipeline |

---

## Configuração e instalação

### Pré-requisitos

- Python 3.11 ou superior
- pip

### Passo 1 — Clone e instale as dependências

```bash
git clone <url-do-repositório>
cd enquadramento-ideologico
pip install -r requirements.txt
```

### Passo 2 — (Opcional) Instale o modelo de linguagem do spaCy

O sistema funciona sem spaCy, mas a lematização fica mais precisa com ele instalado.

```bash
python -m spacy download pt_core_news_sm
```

> Sem o modelo instalado, o sistema usa uma lematização simples (filtragem por tamanho de token). Os resultados ficam um pouco piores, mas tudo funciona normalmente.

---

## Como executar

### Passo 1 — Gerar o corpus de treinamento

```bash
python scripts/generate_corpus.py
```

Cria `data/raw/corpus.jsonl` com 160 textos sintéticos (40 por ideologia).

### Passo 2 — Construir o modelo de referência

```bash
python scripts/build_model.py
```

Processa o corpus, monta o grafo, detecta comunidades e salva o modelo em `outputs/models/model.json`. Precisa ser feito apenas uma vez — ou quando o corpus ou as seeds mudarem.

### Passo 3 — Classificar um texto

```bash
python scripts/classify.py data/examples/exemplo.txt
```

Saída esperada no terminal:

```
DISTRIBUICAO IDEOLOGICA
==================================================
  neoliberal           72.4%  ############################
  conservador          14.1%  #####
  progressista          8.3%  ###
  ancap                 5.2%  ##

Enquadramento dominante: neoliberal (72.4%)
```

Também são geradas duas imagens em `outputs/figures/`:
- `exemplo.png` — subgrafo colorido por ideologia
- `exemplo_bars.png` — gráfico de barras da distribuição

### Métodos de classificação

O método padrão é `graph`, que usa o grafo de co-ocorrência do próprio documento:

```bash
python scripts/classify.py data/examples/exemplo.txt --method graph
```

O método legado `jaccard` considera apenas presença e ausência de termos, ignorando como eles se relacionam no texto:

```bash
python scripts/classify.py data/examples/exemplo.txt --method jaccard
```

**Diferença entre os métodos com `exemplo.txt`:**

```
jaccard  →  neoliberal 70.1%  conservador 11.9%  progressista 11.5%  ancap  6.5%
graph    →  neoliberal 54.9%  conservador 15.0%  progressista 15.0%  ancap 15.0%
```

O método `graph` é mais criterioso: exige que os termos ideológicos co-ocorram nas mesmas janelas de contexto, não apenas que apareçam no texto. Um discurso que usa vocabulário neoliberal de forma isolada e dispersa pontua menos do que um texto em que esses termos se agrupam semanticamente.

### Executar os testes

```bash
pytest
```

---

## Como adicionar mais exemplos

### Adicionar um texto para classificar

Crie um arquivo `.txt` em `data/examples/` com o conteúdo que deseja analisar:

```
data/examples/meu_texto.txt
```

Depois execute:

```bash
python scripts/classify.py data/examples/meu_texto.txt
```

O texto pode ser qualquer coisa — um artigo, um trecho de discurso, um post. O sistema limpa a pontuação e processa automaticamente.

### Adicionar novas palavras-âncora

Edite `data/lexicons/seeds.json`. Cada ideologia tem uma lista de palavras que representam o seu núcleo semântico. Quanto mais representativas forem essas palavras, melhor o modelo ancora as comunidades.

```json
{
  "neoliberal": ["mercado", "privatização", "eficiência", ...],
  "progressista": ["redistribuição", "direitos", "sindicato", ...],
  "conservador": ["família", "tradição", "ordem", ...],
  "ancap": ["liberdade", "propriedade", "voluntário", ...]
}
```

Após editar, reconstrua o modelo:

```bash
python scripts/build_model.py
```

### Adicionar expressões multipalavra

Edite `data/lexicons/markers.txt` — uma expressão por linha:

```
livre mercado
estado mínimo
renda básica
propriedade privada
```

A Trie reconhece essas expressões durante o processamento e as trata como um único token, o que melhora a qualidade do grafo.

### Ajustar os parâmetros do modelo

Edite `config.yaml` para experimentar combinações diferentes:

```yaml
window_size: 5          # quantas palavras por janela de contexto
weight_method: npmi     # frequency | npmi | jaccard
filter_method: kruskal  # kruskal | threshold | disparity
community_method: girvan_newman  # girvan_newman | label_propagation
threshold: 0.1          # peso mínimo por aresta (só usado com filter_method: threshold)
```

Depois reconstrua o modelo para ver o efeito.

---

## Avanços recentes

### Scorer por grafo de co-ocorrência do documento (Fase B)

O classificador original (`jaccard`) tratava o documento como um saco de palavras: comparava apenas quais termos apareciam no texto, sem considerar como eles se relacionavam entre si.

O novo scorer padrão (`graph`) constrói um grafo de co-ocorrência do próprio documento a partir das janelas deslizantes geradas pelo pipeline — a mesma estrutura usada para construir o modelo de referência na Fase A. Para cada ideologia, o score combina:

- **node_score**: centralidade de referência dos termos ideológicos presentes no documento.
- **edge_score**: soma dos pesos das arestas do doc_graph entre pares de termos da mesma ideologia, ponderada pela centralidade de referência de cada ponta.

Isso significa que termos ideológicos que co-ocorrem na mesma janela de contexto contribuem mais do que termos dispersos ao longo do texto. O método está implementado em `src/scoring/doc_graph.py` e `src/scoring/classifier.py`, com 11 novos testes em `tests/test_scoring.py`.

---

## Limitações e potenciais melhorias

### 1. Testes de software

Os testes cobrem as estruturas de dados e os algoritmos individualmente, mas faltam:

- **Testes de integração** que executem o pipeline completo de ponta a ponta com uma entrada real
- **Testes com textos reais** para validar se a classificação faz sentido fora do corpus sintético
- **Casos extremos**: texto vazio, texto com uma única palavra, texto em outro idioma, texto com só stopwords
- A cobertura de código não foi medida; partes do `viz/render.py` e do `model/anchoring.py` não têm nenhum teste direto

### 2. Lógica e algoritmos escolhidos

Co-ocorrência de palavras é uma aproximação superficial da semântica. Os problemas mais sérios:

- **Negação é invisível**: "não privatização" e "defesa da privatização" geram a mesma co-ocorrência após remover stopwords
- **Ironia e contra-discurso**: um texto progressista que critica termos neoliberais pode pontuar para neoliberal, embora o scorer `graph` mitigue parcialmente esse problema ao exigir que os termos co-ocorram entre si
- **Ordem das palavras não existe**: o modelo trata o texto como um saco de palavras dentro de janelas
- O **Girvan-Newman** tem custo O(V · E²) e fica lento para grafos maiores; a alternativa `label_propagation` disponível no código é muito mais rápida
- O número de comunidades (`max_communities`) é ajustado manualmente e afeta muito o resultado final

Uma melhoria direta seria usar embeddings de palavras (Word2Vec, FastText ou BERT) para construir o grafo, pois capturam relações semânticas que co-ocorrência local não captura.

### 3. Vocabulário

O corpus de treinamento é **sintético e gerado por templates** em `scripts/generate_corpus.py`. Isso cria co-ocorrências artificialmente regulares que não existem em textos reais:

- Com textos reais (artigos de jornal, discursos, redes sociais), os padrões são muito mais ruidosos e o modelo atual vai ter desempenho pior
- O vocabulário cobre poucos termos por ideologia e não lida com gírias, neologismos ou variações regionais
- O modelo foi treinado em português formal; textos informais terão desempenho fraco

A melhoria mais impactante seria substituir o corpus sintético por textos reais rotulados (discursos parlamentares, editoriais de jornal, etc.).

### 4. Quantidade de ideologias

O sistema trabalha com apenas 4 ideologias fixas. O espectro político real é mais amplo:

- Ideologias não são categorias discretas; um texto pode ter elementos de várias ao mesmo tempo
- Faltam categorias relevantes no contexto brasileiro: trabalhismo, social-democracia, populismo, etc.
- Adicionar uma nova ideologia exige editar `seeds.json`, adicionar entradas em `generate_corpus.py` (TEMPLATES e LEXICONS) e reconstruir o modelo inteiro

### 5. Coloração e visualização do grafo

A visualização atual em `src/viz/render.py` tem problemas visíveis:

- O layout spring foi implementado manualmente com Fruchterman-Reingold sem as otimizações necessárias; **nós se sobrepõem com frequência** e rótulos ficam ilegíveis quando há muitos termos próximos
- A imagem gerada é estática (PNG); não é possível interagir, aproximar ou arrastar nós
- Em grafos com mais de 30 nós, a visualização fica difícil de interpretar

A biblioteca `pyvis` já está listada nas dependências do projeto e gera visualizações HTML interativas. Substituir a renderização atual por `pyvis` seria uma melhoria direta e de baixo esforço.

### 6. Modelo de referência

O modelo atual tem limitações estruturais importantes:

- É construído a partir de corpus sintético, então **não reflete o uso real da linguagem política**
- Os rótulos de ideologia presentes no `corpus.jsonl` são **completamente ignorados** pelo `build_model.py` — o arquivo lê apenas o campo `"text"`, nunca `"ideology"`. A ancoragem das comunidades é feita unicamente pelas seeds, o que torna o modelo não-supervisionado mesmo tendo dados rotulados disponíveis
- Comunidades sem sobreposição com nenhuma seed ficam rotuladas como `unknown` e são perdidas
- O modelo não é atualizado incrementalmente; qualquer mudança no corpus ou nas seeds exige reconstrução completa

Uma melhoria seria usar os rótulos do `corpus.jsonl` como sinal de treinamento real — por exemplo, construindo grafos separados por ideologia e medindo o peso diferencial de cada aresta entre grupos, em vez de depender unicamente das seeds para ancoragem.