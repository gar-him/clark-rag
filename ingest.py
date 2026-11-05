from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding

# to explore agentic stuff
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context

from config import config_embedding, get_vector_store

import time
import os

def build_index(data_dirs=None, storage_dir="storage"):
    config_embedding()
    # connect to Chroma db
    vector_store = get_vector_store()

    # default path
    if data_dirs is None:
        data_dirs = ["data"]

    loaded_documents = []
    # parse input files and generate list of Document objects
    for directory in data_dirs:
        reader = SimpleDirectoryReader(
            input_dir=directory,
            recursive=True,
            filename_as_id=True,
            required_exts=[".md", ".txt", ".pdf", ".doc"],
        )
        documents = reader.load_data()
        print(f"  {directory}: loaded {len(documents)} documents")
        loaded_documents.extend(documents)

    start = time.time() 

    try: 
        # reload existing index (vector_store, docstore, index_store)
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        # no valid index on disk; buildfresh via full embed 
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(
            loaded_documents, 
            storage_context=storage_context, 
            show_progress=True,
            # BM25 requires node text in docstore
            store_nodes_override=True
        )
    
    # delta check; embeds new/changed docs
    # persist first! 
    index.storage_context.persist(storage_dir)
    refreshed = index.refresh_ref_docs(loaded_documents)
    print(f"  {sum(refreshed)} of {len(refreshed)} documents were new or changed")
    # writes docstore and index_store to disk
    print(f"  docstore now has {len(index.docstore.docs)} nodes")
    end = time.time()
    print(f"Took {end - start:.2f} seconds\n")
        
if __name__ == "__main__":
    build_index()
