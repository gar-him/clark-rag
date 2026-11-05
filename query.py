from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from config import config_embedding, config_llm, get_vector_store

from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine

import time

def get_query_engine():
    # sets models for answer-generation and query embedding
    config_llm()
    config_embedding()

    # connnect to Chroma db, rebuilds index
    vector_store = get_vector_store()
    # index = VectorStoreIndex.from_vector_store(vector_store)
    storage_context = StorageContext.from_defaults(
        persist_dir="storage", vector_store=vector_store
    )
    index = load_index_from_storage(storage_context)

    # define top-k of chunks retrieved per query
    vector_retriever = index.as_retriever(similarity_top_k=8)
    
    nodes = list(index.docstore.docs.values())
    bm25_retriever = BM25Retriever.from_defaults(
        docstore=index.docstore,
        similarity_top_k=5,
    )

    # this top_k actually reaches the LLM, number of citations logged
    retriever = QueryFusionRetriever(
        [vector_retriever, bm25_retriever],
        similarity_top_k=4,
        num_queries=1,
        use_async=True,
        verbose=True,
    )

    return RetrieverQueryEngine.from_args(retriever)

# format and print citation details
def print_citations(source_nodes):
    for node in source_nodes:
        print(node.metadata['file_name'])
        print(node.metadata['file_path'])
        print(f"size: {node.metadata['file_size']}")
        print(f"last modified date: {node.metadata['last_modified_date']}")
        print(f"Score = {node.score:.3f}")
        print(node.text)
        print("\n-------------------- end of citation ---------------------\n")

if __name__ == "__main__":
    query_engine=get_query_engine()
    while True: 
        query = input("> ")
        start = time.time()
        response = query_engine.query(query)
        print_citations(response.source_nodes)
        print(response) # generated response from LLM
        end = time.time() 
        print(f"Took {end - start:.2f} seconds\n")
