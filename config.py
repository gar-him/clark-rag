import os

import chromadb

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.vector_stores.chroma import ChromaVectorStore


# --- Embeddings: stay local -------------------------------------------------
# Embedding models are small enough to run locally (and embedding models aren't available in Ollama Cloud)
DEFAULT_EMBED_MODEL = "nomic-embed-text"
DEFAULT_EMBED_BASE_URL = "http://localhost:11434"

# --- LLM: route through Ollama Cloud ----------------------------------------
# NOTE: "qwen2.5:3b-instruct" was used for testing, sized for my local hardware (NVIDIA GTX 970).
# Cloud model IDs are different — check the current catalog at: https://ollama.com/api/tags

DEFAULT_LLM_MODEL = "gpt-oss:20b"
OLLAMA_CLOUD_HOST = "https://ollama.com"

DEFAULT_REQUEST_TIMEOUT = 360.0
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_COLLECTION_NAME = "quickstart"


def _cloud_client_kwargs():
    """
    Build the auth header llama_index's Ollama wrapper forwards to the
    underlying ollama-python client. Reads the key from the environment
    rather than a literal here, so it never ends up committed to source
    control.
    """
    api_key = os.environ.get("OLLAMA_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OLLAMA_API_KEY is not set. Run `export OLLAMA_API_KEY=your_key` "
            "in your shell before starting this script."
        )
    return {"headers": {"Authorization": f"Bearer {api_key}"}}


def config_embedding(model=DEFAULT_EMBED_MODEL, base_url=DEFAULT_EMBED_BASE_URL):
    Settings.embed_model = OllamaEmbedding(model_name=model, base_url=base_url)


def config_llm(model=DEFAULT_LLM_MODEL, timeout=DEFAULT_REQUEST_TIMEOUT, use_cloud=True):
    if use_cloud:
        Settings.llm = Ollama(
            model=model,
            base_url=OLLAMA_CLOUD_HOST,
            request_timeout=timeout,
            client_kwargs=_cloud_client_kwargs(), # attaches OLLAMA_API_KEY
        )
    else:
        # use_cloud=False and override model to fall back to a local server
        Settings.llm = Ollama(model=model, request_timeout=timeout)


# creates/accesses on-disk Chromadb
def get_vector_store(path=DEFAULT_CHROMA_PATH, name=DEFAULT_COLLECTION_NAME):
    db = chromadb.PersistentClient(path=path)
    chroma_collection = db.get_or_create_collection(name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    return vector_store
