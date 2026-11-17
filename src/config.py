"""Configuração comum aos notebooks Databricks. Sem credenciais AWS no código."""
import hashlib
import os
import tempfile
from pathlib import Path

# Mesmo External Volume utilizado no setup da Fase 3; apenas o prefixo muda.
VOLUME_CATALOG = "workspace"
VOLUME_SCHEMA = "default"
VOLUME_NAME = "vol_trio_drive"
EXTERNAL_VOLUME = f"{VOLUME_CATALOG}.{VOLUME_SCHEMA}.{VOLUME_NAME}"
VOLUME_ROOT = f"/Volumes/{VOLUME_CATALOG}/{VOLUME_SCHEMA}/{VOLUME_NAME}"
PROJECT_PREFIX = "projetos/fiap/tech_challenge_fase4"
PROJECT_S3 = "s3://s3tc2/projetos/fiap/tech_challenge_fase4/"
DEFAULT_DATA_ROOT = f"{VOLUME_ROOT}/{PROJECT_PREFIX}"
NOME_MODELO = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
# Para a primeira execução usa main; a revisão resolvida é registrada no manifesto.
# Depois da execução homologada, substitua main pelo commit registrado.
REVISAO_MODELO = "main"
NOME_COLECAO = "olist_reviews_rag"
DIMENSAO = 384
ARQUIVOS_RAW = [
    "olist_customers_dataset.csv", "olist_geolocation_dataset.csv",
    "olist_order_items_dataset.csv", "olist_order_payments_dataset.csv",
    "olist_order_reviews_dataset.csv", "olist_orders_dataset.csv",
    "olist_products_dataset.csv", "olist_sellers_dataset.csv",
    "product_category_name_translation.csv",
]


def configurar(dbutils):
    """Obtém parâmetros do notebook e cria somente subpastas no Volume existente."""
    parametros = {
        "data_root": os.environ.get("OLIST_DATA_ROOT", DEFAULT_DATA_ROOT),
        "modo_embeddings": "gerar",  # gerar, reusar ou importar_legado
    }
    for nome, padrao in parametros.items():
        try:
            dbutils.widgets.get(nome)
        except Exception:
            dbutils.widgets.text(nome, padrao, nome)
    raiz = dbutils.widgets.get("data_root").strip().rstrip("/")
    if not raiz:
        raise ValueError(
            "Preencha o widget data_root ou DEFAULT_DATA_ROOT em src/config.py. "
            "Use /Volumes/<catalogo>/<schema>/<volume>/<pasta_do_projeto>."
        )
    partes = Path(raiz).parts
    if len(partes) < 6 or partes[1] != "Volumes" or ".." in partes:
        raise ValueError("data_root deve ser uma subpasta de /Volumes/catalogo/schema/volume.")
    volume = Path(*partes[:5])
    if not volume.is_dir():
        raise FileNotFoundError(f"Volume não acessível: {volume}. Confira compute, nome e permissões.")
    modo = dbutils.widgets.get("modo_embeddings").strip()
    if modo not in {"gerar", "reusar", "importar_legado"}:
        raise ValueError("modo_embeddings: use gerar, reusar ou importar_legado.")
    raiz = Path(raiz)
    cfg = {
        "base": raiz, "raw": raiz / "data/raw", "processed": raiz / "data/processed",
        "reports": raiz / "reports", "modo_embeddings": modo,
    }
    for campo in ("raw", "processed", "reports"):
        cfg[campo].mkdir(parents=True, exist_ok=True)
    cfg["documentos"] = cfg["processed"] / "olist_reviews_documentos_rag.jsonl"
    cfg["embeddings"] = cfg["processed"] / "olist_reviews_embeddings.npy"
    cfg["manifesto"] = cfg["processed"] / "olist_reviews_embeddings.manifest.json"
    # SQLite/HNSW operam em disco POSIX local; nunca diretamente no Volume/S3.
    namespace = hashlib.sha256(str(raiz).encode()).hexdigest()[:16]
    cfg["chroma_local"] = Path(tempfile.gettempdir()) / "olist_rag" / namespace
    cfg["chroma_local"].mkdir(parents=True, exist_ok=True)
    os.environ["OLIST_DATA_ROOT"] = str(raiz)
    print("Dados persistentes:", raiz)
    print("Modo de embeddings:", modo)
    print("Cache Chroma temporário:", cfg["chroma_local"])
    return cfg
