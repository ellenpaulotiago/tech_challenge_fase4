import os
from pathlib import Path

import chromadb
from sentence_transformers import SentenceTransformer


# Configurações do projeto.
BASE_DIR = Path(__file__).resolve().parent.parent

# Usar o mesmo armazenamento configurável do notebook.
caminho_configurado = os.environ.get("OLIST_CHROMA_DIR")
diretorio_local_app = os.environ.get("LOCALAPPDATA")

if caminho_configurado:
    DIRETORIO_BANCO_VETORIAL = Path(caminho_configurado).expanduser().resolve()
elif diretorio_local_app:
    DIRETORIO_BANCO_VETORIAL = (
        Path(diretorio_local_app) / "OlistRAG" / "vector_store"
    ).resolve()
else:
    raise EnvironmentError(
        "Defina OLIST_CHROMA_DIR para localizar o banco vetorial."
    )

NOME_MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

NOME_COLECAO = "olist_reviews_rag"


def inicializar_busca_vetorial():

    # Carregar o mesmo modelo utilizado na geração dos embeddings.
    modelo = SentenceTransformer(NOME_MODELO)

    # Conectar ao banco vetorial persistente.
    cliente = chromadb.PersistentClient(
        path=str(DIRETORIO_BANCO_VETORIAL)
    )

    # Acessar a coleção existente, sem recriá-la.
    colecao = cliente.get_collection(
        name=NOME_COLECAO
    )

    return modelo, colecao


def buscar_avaliacoes(
    pergunta: str,
    modelo,
    colecao,
    quantidade: int = 5
) -> list[dict]:

    # Validar os parâmetros recebidos.
    if not pergunta.strip():
        raise ValueError("A pergunta não pode estar vazia.")

    if quantidade < 1:
        raise ValueError("A quantidade deve ser maior que zero.")

    # Gerar o embedding da pergunta.
    embedding_pergunta = modelo.encode(
        pergunta,
        convert_to_numpy=True
    )

    # Consultar o banco vetorial.
    resultados = colecao.query(
        query_embeddings=[embedding_pergunta.tolist()],
        n_results=quantidade,
        include=["documents", "metadatas", "distances"]
    )

    # Validar os campos retornados.
    assert resultados["documents"] is not None
    assert resultados["metadatas"] is not None
    assert resultados["distances"] is not None

    # Organizar os documentos recuperados.
    avaliacoes_recuperadas = []

    for indice, documento_id in enumerate(resultados["ids"][0]):

        avaliacao = {
            "documento_id": documento_id,
            "texto": resultados["documents"][0][indice],
            "metadados": resultados["metadatas"][0][indice],
            "distancia": resultados["distances"][0][indice]
        }

        avaliacoes_recuperadas.append(avaliacao)

    return avaliacoes_recuperadas
