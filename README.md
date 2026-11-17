# Olist Voice of Customer Intelligence — Tech Challenge Fase 4

Projeto de **Voice of Customer Intelligence** baseado em **Retrieval-Augmented Generation (RAG)** para análise inteligente de avaliações de clientes do e-commerce Olist. Desenvolvido como entregável principal do **Tech Challenge — Fase 4** da pós-graduação FIAP / PosTech em Inteligência Artificial Generativa, em parceria com a Databricks Academy (*Generative AI Engineering Pathway*).


## 🎯 1. Visão Geral e Problema de Negócio

Empresas de e-commerce recebem diariamente milhares de avaliações contendo percepções, elogios, reclamações e relatos sobre diferentes etapas da jornada de compra. Na base **Brazilian E-Commerce Public Dataset by Olist** (disponível no Kaggle com cerca de 100 mil pedidos entre 2016 e 2018), grande parte dos insights mais valiosos encontra-se em **comentários textuais não estruturados**.

Analistas e gestores enfrentam dificuldades para analisar manualmente esse volume massivo de opiniões para responder a perguntas estratégicas como:
- *Quais são os principais motivos de insatisfação dos clientes?*
- *Quais reclamações sobre entrega aparecem com maior frequência?*
- *Quais evidências concretas nos comentários sustentam determinado insight?*

### A Solução
O **Olist RAG** transforma o conjunto de comentários das avaliações em uma **base de conhecimento inteligente e explorável**. A solução recebe perguntas em linguagem natural em português, realiza busca vetorial por similaridade semântica nos relatos dos clientes e utiliza o LLM **Meta Llama 3.3 70B Instruct** para sintetizar respostas **estritamente fundamentadas em evidências**, citando cada fonte utilizada e abstendo-se de responder quando não há dados suficientes ou a pergunta está fora do escopo.

---

## 🏗 2. Arquitetura da Solução e Fluxo RAG

O pipeline foi projetado para execução no **Databricks** integrando armazenamento S3 via Volumes externos, indexação vetorial no ChromaDB e inferência via endpoint gerenciado.

```text
  [ Pergunta do Usuário em Linguagem Natural ]
                       │
                       ▼
  [ Vetorização da Pergunta (Sentence-Transformers MiniLM) ]
                       │
                       ▼
  [ Busca Vetorial (ChromaDB - Até 12 documentos candidatos) ]
                       │
                       ▼
  [ Construção do Contexto (Top 10 documentos / max 12.000 caracteres) ]
                       │
                       ▼
  [ Inferência LLM (Databricks Meta Llama 3.3 70B Instruct) ]
                       │
                       ▼
  [ Resposta Fundamentada com Citações de Evidências (ou Abstenção) ]
```

### Componentes Tecnológicos:
- **Plataforma de Execução**: Databricks (Compute Standard v6, Python 3.12, CPU).
- **Armazenamento de Dados**: AWS S3 integrado via Volume Databricks (`/Volumes/workspace/default/vol_trio_drive/projetos/fiap/tech_challenge_fase4/`).
- **Modelo de Embeddings**: `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (vetores densos de 384 dimensões).
- **Banco Vetorial**: `ChromaDB v1.0.20` (indexação em memória/temporária com suporte a documentos JSONL e embeddings `.npy`).
- **Modelo de Linguagem (LLM)**: Endpoint gerenciado Databricks `databricks-meta-llama-3-3-70b-instruct` (*Meta Llama 3.3 70B Instruct*) acessado via Databricks SDK / OpenAI Client (`response_format` em `json_schema`).

---

## 📋 3. Atendimento aos Requisitos do Desafio

### 1. Construção da Base de Conhecimento
- **Processamento e Seleção**: Da base original de **99.224 avaliações**, a etapa de tratamento filtrou e organizou **42.687 documentos com texto útil** (descartando registros vazios ou sem comentário).
- **Estruturação**: Gerado o arquivo `data/processed/olist_reviews_documentos_rag.jsonl` contendo comentários e metadados das avaliações.
- **Embeddings e Indexação**: Embeddings de 384 dimensões pré-calculados e salvos em `data/processed/olist_reviews_embeddings.npy` alinhados 1:1 com os documentos, permitindo inicializar a coleção no ChromaDB no ambiente temporário de execução sem recalcular vetores.

### 2. Implementação do Pipeline RAG
- **Busca Semântica**: O módulo `src/busca_vetorial_databricks.py` recebe a consulta, gera a representação vetorial e recupera até 12 relatos candidatos por similaridade semântica.
- **Construção de Contexto**: O módulo `src/rag_pipeline_databricks.py` seleciona até 10 documentos (máximo de 12.000 caracteres) e monta o prompt estruturado com rótulos de fonte (ex: `E1`, `E2`, ..., `E10`).
- **Geração Estruturada**: O LLM gera respostas em formato JSON estruturado (`json_schema`), definindo o status (`respondido`, `fora_escopo`, `evidencias_insuficientes`), as afirmações e as fontes associadas.

### 3. Respostas Fundamentadas em Evidências e Rastreabilidade
- As respostas ligam cada afirmação gerada diretamente ao comentário original do cliente que a sustenta.
- *Exemplo*: Na pergunta `Q01` sobre atrasos na entrega, a resposta vincula a afirmação sobre impacto na obra à fonte **E4** (*"O atraso na entrega prejudicou a obra"*).

### 4. Tratamento de Abstenção e Limitações do Modelo
Para evitar alucinações e garantir a confiabilidade do sistema:
- **Fora do Escopo**: Perguntas desalinhadas do domínio (ex: `Q17` — *"Qual a previsão do tempo para amanhã em Guarulhos?"*) encerram a execução no status `fora_escopo` sem emitir afirmações.
- **Evidências Insuficientes**: Perguntas que exigem contagens globais ou porcentagens não calculáveis por amostragem vetorial (ex: `Q21` — *"Qual o percentual de reclamações de atraso em toda a base?"*) identificam relatos de atraso, mas sinalizam `evidencias_insuficientes` para calcular a taxa percentual global.

---

### 4. Olist RAG — Tech Challenge Fase 4
O projeto responde, em português, a perguntas sobre experiências de compra registradas nas avaliações do  **Olist Brazilian E-Commerce** . A busca vetorial recupera comentários da base; o modelo de linguagem seleciona as evidências e gera afirmações com fontes. Se a pergunta for externa ao domínio ou exigir dados que os relatos recuperados não fornecem, o sistema se abstém.
Na rodada de  **7 de novembro de 2026** , o pipeline no Databricks executou  **24 de 24 perguntas** , acertou os  **24 status esperados**  e produziu  **13 respostas completas entre as 16 geradas** .
#### Ambiente e arquivos
O  **Databricks**  executa os notebooks. O  **Git**  versiona o código e a documentação na branch dev. O  **S3** , acessado pelo Volume externo workspace.default.vol_trio_drive, armazena os dados e os resultados.
```text
Raiz no Databricks:
  /Volumes/workspace/default/vol_trio_drive/projetos/fiap/tech_challenge_fase4/
Prefixo correspondente no S3:
  s3://s3tc2/projetos/fiap/tech_challenge_fase4/

```
| Caminho relativo à raiz do Volume | Conteúdo |
| ------ | ------ |
| data/raw/ | CSVs originais da Olist. |
| data/processed/olist_reviews_documentos_rag.jsonl | Comentários e metadados das avaliações. |
| data/processed/olist_reviews_embeddings.npy | Vetores de 384 dimensões alinhados aos documentos do JSONL. |
| src/busca_vetorial_databricks.py | Módulo de recuperação exportado pelo Notebook 03. |
| src/rag_pipeline_databricks.py | Módulo do pipeline exportado pelo Notebook 04. |
| reports/notebook04/ | Registros JSON das consultas individuais. |
| reports/notebook05_databricks/ | Plano, manifestos, resultados por pergunta e métricas de avaliação. |

Abra a branch dev na pasta Git do workspace. Use a identidade do Databricks com acesso de leitura e escrita ao Volume e permissão de consulta ao endpoint de chat. Os notebooks 04 e 05 carregam os módulos pelo Volume. Se os módulos também forem versionados no Git, mantenha as duas cópias alinhadas.
##### Dependências dos Notebooks 04 e 05
A execução validada usou o ambiente  **Standard v6, Python 3.12, CPU** , com as versões abaixo. Em uma sessão nova, instale as dependências antes de carregar o pipeline. Se a sessão já contém as versões validadas, siga para a configuração dos caminhos.
| Pacote | Versão registrada |
| ------ | ------ |
| PyTorch | 2.6.0+cpu |
| Transformers / Sentence Transformers | 4.57.6 / 5.1.0 |
| ChromaDB | 1.0.20 |
| Databricks SDK / OpenAI | 0.122.0 / 1.109.1 |
| NumPy / pandas | 2.3.4 / 2.3.3 |

Se o notebook ainda contiver a instalação antiga com --upgrade, substitua essa célula antes da execução. A atualização ampla gerou conflito com python-dateutil durante a preparação da rodada. Os comandos abaixo são uma referência para reconstituir as versões diretas registradas e usar a distribuição CPU do PyTorch. O conjunto de dependências transitivas não está congelado. Execute cada comando em uma célula própria, preservando o SDK e as bibliotecas base nas versões acima.
```python
%pip install "torch==2.6.0" --index-url https://download.pytorch.org/whl/cpu

```
```python
%pip install "torch==2.6.0" "transformers==4.57.6" "sentence-transformers==5.1.0" "chromadb==1.0.20" "openai==1.109.1" "python-dateutil==2.9.0.post0" "packaging==24.0" "six==1.16.0"

```
```python
dbutils.library.restartPython()

```
```python
%pip check

```
Confira No broken requirements found, as versões e os imports antes de prosseguir. Em outro ambiente base, a lista de dependências pode ser diferente; se houver conflito, verifique o pacote indicado antes de executar o RAG. O cliente get_open_ai_client() funcionou na rodada registrada e emitiu aviso de descontinuação. Uma troca de cliente deve ser testada antes de uma nova rodada comparável.
#### Ordem de execução
| Notebook | O que executar e conferir |
| ------ | ------ |
| **00 — Configuração do ambiente** | Valide o Volume externo e a raiz do projeto. Confira as pastas e disponibilize os CSVs originais em data/raw/ se for reconstruir a base. |
| **01 — Exploração dos dados** | Confira a estrutura dos arquivos e a presença de títulos, comentários e metadados. |
| **02 — Preparação da base de conhecimento** | Gere ou valide o JSONL. A preparação original registrou  **42.687 documentos com texto útil**  entre  **99.224 avaliações** . |
| **03 — Embeddings e busca vetorial** | Valide o alinhamento entre JSONL e NPY e teste a busca no ChromaDB. Execute a exportação de busca_vetorial_databricks.py. Preserve o par de arquivos da rodada se o objetivo for reproduzir a configuração avaliada. |
| **04 — Integração do LLM com o RAG** | Valide os caminhos, prepare a coleção, configure o endpoint, execute os testes e faça as consultas da demonstração. Execute a célula final de exportação de rag_pipeline_databricks.py para o 05. |
| **05 — Avaliação** | Confira o plano, execute os lotes, releia os CSVs preenchidos e exporte as métricas e o relatório. |

O modelo de embeddings é sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2. Nos notebooks de integração e avaliação, o índice do Chroma pode ser reconstruído no armazenamento temporário do compute a partir do JSONL e do NPY, sem recalcular os vetores da base. A configuração da rodada recuperou até 12 candidatos e enviou ao LLM até 10 documentos e 12.000 caracteres. A seleção identifica fontes por rótulos; o Python recupera os trechos originais e a geração cita apenas as fontes selecionadas.
##### Configuração do endpoint
Na célula de configuração do 04, use os valores abaixo. No 05, use os mesmos campos de modelo e controle a inferência por EXECUTAR_AVALIACAO.
```python
LLM_ENDPOINT = "databricks-meta-llama-3-3-70b-instruct"
LLM_MODELO_VERSAO = "Meta Llama 3.3 70B Instruct; endpoint gerenciado Databricks"
EXECUTAR_CONSULTA = False

```
A listagem de endpoints do 04 verifica quais opções aparecem no workspace. Antes de habilitar a consulta real, confirme que o endpoint escolhido aceita uma solicitação curta de chat com response_format do tipo json_schema. A identificação em LLM_MODELO_VERSAO é um registro descritivo; não fixa uma revisão imutável do modelo.
##### Nova rodada no Notebook 05
1. Com EXECUTAR_AVALIACAO = False e EXECUCAO_ID = "", execute a preparação. Coloque o plano perguntas_avaliacao_v1.csv em reports/notebook05_databricks/planos/. A célula de leitura deve mostrar  **24 referências confirmadas** .
2. Confirme o carregamento do módulo RAG na Etapa 3. Se a célula que registra o módulo em sys.modules precisar do import, inclua import sys nela.
3. Defina EXECUTAR_AVALIACAO = True e MAX_PERGUNTAS_POR_EXECUCAO = 3. Deixe o ID vazio somente para o primeiro lote. Após a execução, guarde o ID exibido e informe-o na configuração antes de reexecutar a célula do lote. Continue até obter  **24/24** . Ao retomar, preserve os valores exatos do manifesto, inclusive o texto de LLM_MODELO_VERSAO. Se os dados, o código, o modelo ou o plano mudarem, inicie outra rodada.
4. Na Etapa 5, preserve os nomes, chaves e campos de evidência de revisao_recuperacao.csv, revisao_afirmacoes.csv e revisao_respostas.csv. Preencha apenas os campos de avaliação e justificativa. Para respostas abstidas, completude e clareza ficam vazias.
5. Para reler uma rodada já gravada, informe seu ID, mantenha EXECUTAR_AVALIACAO = False e execute a célula que carrega os registros. Releia os CSVs na Etapa 5 e execute as Etapas 6 e 7. Esse caminho recalcula os relatórios sem repetir as chamadas ao LLM.
#### Resultados da avaliação
**Rodada:**  20261107T203256Z_3de570ec.
| Indicador | Resultado |
| ------ | ------ |
| Cobertura de execução | 24/24 |
| Acerto de status | 24/24 |
| Respostas / fora do escopo / evidências insuficientes | 16 / 5 / 3 |
| Abstenções com status exato | 8/8 |
| Erros técnicos | 0 |
| Precisão média nos primeiros 3 / 5 / 10 resultados, em 16 perguntas respondíveis | 87,50% / 82,50% / 79,38% |
| Afirmações com citação estrutural válida | 72/72 |
| Afirmações classificadas como sustentadas | 72/72 |
| Respostas completas / claras | 13/16 / 16/16 |
| Latência mediana / p95 | 4,72 s / 11,23 s |
| Consumo reportado | 92.485 tokens em 40 chamadas |

Os arquivos ficam em reports/notebook05_databricks/execucoes/20261107T203256Z_3de570ec/. Preserve o manifesto, os resultados por pergunta e os três CSVs preenchidos. A Etapa 7 exportou metricas.json, comparacao_status.csv, precision_por_pergunta.csv, auditoria_citacoes.csv, resumo_execucoes.csv e resumo_avaliacao.md.
##### Limitações observadas
A busca por similaridade retorna relatos próximos à pergunta, sem garantir relevância de todos os resultados. Citações estruturalmente válidas identificam documentos e trechos rastreáveis; o suporte da afirmação e a adequação à pergunta são critérios distintos.  **O acerto de status foi de 100% nesta rodada; a completude ficou em 81,25%.**
*   **Q06 — Qualidade:**  a resposta omitiu um relato negativo recuperado.
*   **Q10 — Quantidade:**  dois relatos descreviam itens a menos, mas a resposta incluiu três exemplos sobre valor, produto diferente ou medida.
*   **Q11 — Embalagem:**  a resposta incluiu um relato de produto amassado, que não estabelece dano na embalagem.
A amostra recuperada não permite estimar prevalência, percentuais globais ou rankings da base. Recall e custo monetário não foram estimados. O resultado descreve este plano e esta configuração, sem garantir a mesma qualidade para perguntas novas.
#### Roteiro de demonstração
**Duração sugerida: 3 minutos.**  Abra o Notebook 04 na Etapa 6 e deixe os três JSONs gravados disponíveis para exibição. Para demonstração ao vivo, prepare a coleção antes de gravar, defina EXECUTAR_CONSULTA = True e altere PERGUNTA_RAG para um caso por vez. Reexecute a configuração e as células de consulta, apresentação e salvamento. As saídas ao vivo podem variar; os arquivos abaixo preservam o que foi observado.
| Caso e pergunta | O que mostrar na tela | Fala sugerida |
| ------ | ------ | ------ |
| **Q01:**  O que os clientes relatam sobre atraso na entrega? | respondido, duas chamadas e cinco afirmações. Abra a fonte  **E4**  e mostre o trecho “O atraso na entrega prejudicou a obra”. | “A resposta liga cada afirmação ao comentário que a sustenta. Aqui vemos o impacto relatado por um cliente.” |
| **Q17:**  Qual é a previsão do tempo para amanhã em Guarulhos? | fora_escopo, uma chamada, sem afirmações ou fontes utilizadas. | “Previsão do tempo está fora do domínio das avaliações de compra. O fluxo se encerra na seleção.” |
| **Q21:**  Qual o percentual de reclamações de atraso em toda a base? | evidencias_insuficientes, uma chamada, sem afirmações ou fontes utilizadas. | “Atraso faz parte do domínio. Há relatos recuperados, mas eles não fornecem os totais necessários para calcular um percentual da base inteira.” |

Os três registros do Notebook 04, em reports/notebook04/, são:
| Caso | JSON gravado |
| ------ | ------ |
| Q01 | consulta_d6506154b79c4c74a2e99da5b17c11f2.json |
| Q17 | consulta_e3092364a9194db28bc756387a619598.json |
| Q21 | consulta_013e652c40c54d08bf6c83d91dfd5fa6.json |

Feche a demonstração no resumo_avaliacao.md da rodada do Notebook 05. Mostre os  **24 status corretos** , as  **oito abstenções corretas**  e a diferença entre resposta sustentada e resposta completa, usando a Q10 como exemplo.

## 5. Diretório do Projeto
```text
tech-challenge-fase4/
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   ├── 00_configuracao_ambiente.ipynb
│   ├── 01_exploracao_dados.ipynb
│   ├── 02_preparacao_base_conhecimento.ipynb
│   ├── 03_embeddings_busca_vetorial.ipynb
│   ├── 04_integracao_llm_rag_databricks.ipynb
│   └── 05_avaliacao_rag_databricks.ipynb
│
├── src/
│   ├── busca_vetorial_databricks.py
│   ├── rag_pipeline_databricks.py
│   ├── __init__.py
│   ├── artefatos.py
│   ├── busca_vetorial.py
│   ├── config.py
│   │
│   └── __pycache__/
│       └── rag_pipeline_databricks.cpython-312.pyc
│
├── config/
│   └── config.json
│
├── docs/
│   ├── mapa_progresso_olist.png
│   ├── olist_data_schema.png
│   ├── pipeline_rag_olist.png
│   ├── roadmap_projeto.png
│   └── TechChallenge04_OlistRAG_Passos_Realizados_Databricks.docx
│
├── data/
│   ├── processed/
│   │   ├── olist_reviews_documentos_rag.jsonl
│   │   ├── olist_reviews_embeddings.part1.rar
│   │   ├── olist_reviews_embeddings.part2.rar
│   │   └── olist_reviews_embeddings.part3.rar
│   │
│   └── raw/
│       ├── olist_customers_dataset.csv
│       ├── olist_geolocation_dataset.zip
│       ├── olist_order_items_dataset.csv
│       ├── olist_order_payments_dataset.csv
│       ├── olist_order_reviews_dataset.csv
│       ├── olist_order_dataset.csv
│       ├── olist_products_dataset.csv
│       ├── olist_sellers_dataset.csv
│       └── product_category_name_translation.csv
│
└── reports/
    ├── 00_ambiente.json
    ├── 00_inventario_raw.json
    ├── 01_exploracao.json
    ├── 02_preparacao.json
    │
    └── avaliacao/
        ├── perguntas_avaliacao_v1.csv
        │
        ├── demonstracao/
        │   ├── Q01.json
        │   ├── Q17.json
        │   └── Q21.json
        │
        └── 20261107T203256Z_3de570ec/
            ├── manifesto_execucao.json
            ├── revisao_recuperacao.csv
            ├── revisao_afirmacoes.csv
            ├── revisao_respostas.csv
            ├── metricas.json
            ├── comparacao_status.csv
            ├── precision_por_pergunta.csv
            ├── auditoria_citacoes.csv
            ├── resumo_execucoes.csv
            ├── resumo_avaliacao.md
            │
            └── resultados/
                └── [JSONs individuais das 24 perguntas]
```
