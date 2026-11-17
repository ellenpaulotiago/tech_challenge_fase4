"""Recuperação Chroma reconstruível a partir dos artefatos no Volume."""
from pathlib import Path

import numpy as np

from .artefatos import carregar_documentos, carregar_embeddings_validados, sha256
from .config import NOME_MODELO, NOME_COLECAO


def reconstruir_colecao(cfg, df, vetores):
    import chromadb
    # A revisão dos artefatos faz parte do diretório. Alterações não reutilizam índice antigo.
    chave = sha256(cfg["manifesto"])
    cliente = chromadb.PersistentClient(path=str(Path(cfg["chroma_local"]) / chave))
    colecao = cliente.get_or_create_collection(
        name=NOME_COLECAO, embedding_function=None,
        configuration={"hnsw": {"space": "cosine"}},
        metadata={"manifesto_sha256": chave},
    )
    if (colecao.metadata or {}).get("manifesto_sha256") != chave:
        raise ValueError("Coleção pertence a outra versão dos artefatos.")
    lote = min(512, cliente.get_max_batch_size())
    metadados = [{"review_id": str(r.review_id), "order_id": str(r.order_id),
                 "review_score": int(r.review_score), "review_creation_date": str(r.review_creation_date)}
                for r in df.itertuples(index=False)]
    ids = df.documento_id.tolist()
    textos = df.texto_documento.tolist()
    # Upsert permite retomar uma inserção interrompida sem duplicar os IDs.
    for inicio in range(0, len(df), lote):
        fim = inicio + lote
        colecao.upsert(ids=ids[inicio:fim], documents=textos[inicio:fim],
                       metadatas=metadados[inicio:fim], embeddings=vetores[inicio:fim].tolist())
    if colecao.count() != len(df):
        raise ValueError("Contagem da coleção não corresponde aos documentos.")
    return cliente, colecao


def inicializar_busca_vetorial(cfg, modelo=None):
    """Carrega modelo e reconstrói o cache; não depende da sessão do Notebook 03."""
    from sentence_transformers import SentenceTransformer
    df = carregar_documentos(cfg["documentos"])
    vetores, manifesto = carregar_embeddings_validados(cfg, df, NOME_MODELO)
    if modelo is None:
        modelo = SentenceTransformer(NOME_MODELO, revision=manifesto["revisao"])
    _, colecao = reconstruir_colecao(cfg, df, vetores)
    return modelo, colecao


def buscar_avaliacoes(pergunta, modelo, colecao, quantidade=5, filtros=None):
    if not isinstance(pergunta, str) or not pergunta.strip():
        raise ValueError("A pergunta deve ser uma string não vazia.")
    if isinstance(quantidade, bool) or not isinstance(quantidade, int) or quantidade < 1:
        raise ValueError("A quantidade deve ser um inteiro positivo.")
    if colecao.count() == 0:
        return []
    vetor = modelo.encode(pergunta.strip(), convert_to_numpy=True, normalize_embeddings=False)
    if not np.isfinite(vetor).all() or np.linalg.norm(vetor) == 0:
        raise ValueError("Embedding da pergunta inválido.")
    parametros = dict(query_embeddings=[vetor.tolist()], n_results=min(quantidade, colecao.count()),
                      include=["documents", "metadatas", "distances"])
    if filtros is not None:
        parametros["where"] = filtros
    r = colecao.query(**parametros)
    return [{"documento_id": identificador, "texto": r["documents"][0][i],
             "metadados": r["metadatas"][0][i], "distancia": float(r["distances"][0][i])}
            for i, identificador in enumerate(r["ids"][0])]
