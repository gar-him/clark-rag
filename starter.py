import asyncio
import time
from llama_index.core import VectorStoreIndex, Settings
from llama_index.core.agent.workflow import FunctionAgent, AgentWorkflow
from llama_index.core.workflow import Context
from config import config_embedding, config_llm, get_vector_store

# embeddings stay local (Ollama Cloud doesn't serve embedding models)
config_embedding()
# LLM routes through Ollama Cloud (reads OLLAMA_API_KEY from env, defaults
# to config.py's DEFAULT_LLM_MODEL — check https://ollama.com/api/tags
# for the current cloud catalog if you want a different model)
config_llm()

# reuse the corpus already embedded and persisted by build.py, instead of
# re-reading "data" and re-embedding everything from scratch each run
vector_store = get_vector_store()
index = VectorStoreIndex.from_vector_store(vector_store)
query_engine = index.as_query_engine(similarity_top_k=2)

def multiply(a: float, b: float) -> float:
    """Useful for finding product of two numbers."""
    print(f"mult called with a={a}, b={b}")
    return a * b

def add(a: float, b: float) -> float:
    """Useful for finding sum of two numbers."""
    print(f"add called with a={a}, b={b}")
    return a+b

async def search_documents(query: str) -> str:
    """Useful for answering natural language questions about Ben Franklin, cities, and the Clippers report by Watchtell Lipton."""
    print(f"search_documents called with query={query}")
    response = await query_engine.aquery(query)
    print(f"search_documents raw result: {response}")
    return str(response)

agent = FunctionAgent(
    tools=[multiply, add, search_documents],
    llm = Settings.llm,
system_prompt="You are a helpful assistant that can access data about the Clippers and Kawhi Leonard.",
)

ctx = Context(agent)
async def main():
    start = time.time()
    response = await agent.run("Who is Kawhi Leonard, and why might he be controversial?", ctx=ctx)
    end = time.time()
    print(str(response))
    print(f"Took {end - start:.2f} seconds")
    start = time.time()
    response = await agent.run("What is a good city for food and d&d culture?", ctx=ctx)
    end = time.time()
    print(str(response))
    print(f"Took {end - start:.2f} seconds")

if __name__ == "__main__":
    asyncio.run(main())
