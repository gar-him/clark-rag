
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.ollama import OllamaEmbedding
# to explore agentic embedding
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context

from config import config_embedding, get_vector_store

import time
import os

def build_index(data_dir="data", storage_dir="storage"):
    config_embedding()
    # connect to Chroma db
    vector_store = get_vector_store()

    # parse input files and generate list of Document objects
    reader = SimpleDirectoryReader(input_dir=data_dir, recursive=True, filename_as_id=True)
    documents = reader.load_data()

    start = time.time() 

    try: 
        # reload existing index (vector_store, docstore, index_store)
        storage_context = StorageContext.from_defaults(persist_dir=storage_dir, vector_store=vector_store)
        index = load_index_from_storage(storage_context)
    except FileNotFoundError:
        # no valid index on disk; buildfresh via full embed 
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        index = VectorStoreIndex.from_documents(documents, storage_context=storage_context)
    
    # re-embeds new/changed docs
    index.refresh_ref_docs(documents)
    # writes docstore and index_store to disk
    index.storage_context.persist(storage_dir)
    end = time.time()
    print(f"Took {end - start:.2f} seconds\n")
        
if __name__ == "__main__":
    build_index()
