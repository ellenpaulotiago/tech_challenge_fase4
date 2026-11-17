"""Contratos e persistência para manter documentos e vetores alinhados."""
import hashlib
import importlib.metadata
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

COLUNAS = ["documento_id", "review_id", "order_id", "review_score",
           "review_creation_date", "texto_documento"]


def sha256(arquivo):
    h = hashlib.sha256()
    with Path(arquivo).open("rb") as stream:
        for bloco in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(bloco)
    return h.hexdigest()


def hash_documentos(df):
    # Inclui conteúdo, IDs, metadados e ordem, sem depender do espaçamento do JSONL.
    registros = df[COLUNAS].to_dict("records")
    dados = json.dumps(registros, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(dados.encode("utf-8")).hexdigest()


def validar_documentos(df):
    if not set(COLUNAS).issubset(df.columns) or df.empty:
        raise ValueError("Base vazia ou sem as seis colunas obrigatórias.")
    if df[COLUNAS].isna().any().any():
        raise ValueError("Há valores ausentes nos documentos.")
    for campo in ["documento_id", "review_id", "order_id", "review_creation_date", "texto_documento"]:
        if not df[campo].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"Campo {campo} com tipo inválido ou vazio.")
    if df.documento_id.duplicated().any() or df.duplicated(["review_id", "order_id"]).any():
        raise ValueError("Identificadores duplicados na base.")
    ids = [json.dumps([r, o], ensure_ascii=False, separators=(",", ":"))
           for r, o in zip(df.review_id, df.order_id)]
    if ids != df.documento_id.tolist():
        raise ValueError("documento_id não corresponde ao par review_id e order_id.")
    if not df.review_score.isin([1, 2, 3, 4, 5]).all():
        raise ValueError("Notas fora da escala inteira 1 a 5.")
    if pd.to_datetime(df.review_creation_date, errors="coerce").isna().any():
        raise ValueError("Datas de avaliação inválidas.")


def carregar_documentos(arquivo):
    df = pd.read_json(arquivo, lines=True, convert_dates=False,
                      dtype={k: "str" for k in COLUNAS if k != "review_score"})
    validar_documentos(df)
    return df


def validar_embeddings(vetores, quantidade, dimensao=384):
    if vetores.shape != (quantidade, dimensao) or vetores.dtype != np.float32:
        raise ValueError(f"Embeddings devem ser float32 com formato {(quantidade, dimensao)}.")
    if not np.isfinite(vetores).all() or (np.linalg.norm(vetores, axis=1) == 0).any():
        raise ValueError("Embeddings contêm valores não finitos ou vetores nulos.")


def ambiente():
    resultado = {"python": platform.python_version()}
    for nome in ["pandas", "numpy", "chromadb", "sentence-transformers", "torch", "transformers"]:
        try:
            resultado[nome] = importlib.metadata.version(nome)
        except importlib.metadata.PackageNotFoundError:
            resultado[nome] = "não instalado"
    return resultado


def registrar_manifesto(cfg, df, vetores, modelo, revisao, origem):
    validar_documentos(df)
    validar_embeddings(vetores, len(df))
    manifesto = {
        "schema_version": 1, "criado_em_utc": datetime.now(timezone.utc).isoformat(),
        "modelo": modelo, "revisao": revisao, "normalizar": False,
        "quantidade": len(df), "dimensao": 384, "dtype": "float32",
        "hash_documentos_ordenados": hash_documentos(df),
        "sha256_jsonl": sha256(cfg["documentos"]),
        "sha256_npy": sha256(cfg["embeddings"]), "origem": origem,
        "ambiente": ambiente(),
    }
    cfg["manifesto"].write_text(json.dumps(manifesto, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifesto


def carregar_embeddings_validados(cfg, df, modelo):
    if not cfg["manifesto"].is_file():
        raise FileNotFoundError("NPY sem manifesto: use gerar ou importar_legado no Notebook 03.")
    m = json.loads(cfg["manifesto"].read_text(encoding="utf-8"))
    if m.get("schema_version") != 1 or m.get("modelo") != modelo or m.get("normalizar") is not False:
        raise ValueError("Modelo ou configuração incompatível com o manifesto.")
    if m["hash_documentos_ordenados"] != hash_documentos(df):
        raise ValueError("Documentos ou ordem mudaram. Gere novos embeddings.")
    if m["sha256_npy"] != sha256(cfg["embeddings"]):
        raise ValueError("O NPY difere do arquivo registrado no manifesto.")
    vetores = np.load(cfg["embeddings"], allow_pickle=False)
    validar_embeddings(vetores, len(df))
    return vetores, m
