from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, StorageContext, load_index_from_storage
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context
from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from config import config_embedding, config_llm, get_vector_store

import time

def get_query_engine():
    # sets models for answer-generation and query embedding
    config_llm()
    config_embedding()

    # connnect to Chroma db, rebuilds index
    vector_store = get_vector_store()
    index = VectorStoreIndex.from_vector_store(vector_store)
    # define top-k of chunks retrieved per query
    return index.as_query_engine(similarity_top_k=2)

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
