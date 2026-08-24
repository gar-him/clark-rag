## Setup

1. Clone the repo
2. `python -m venv .venv && source .venv/bin/activate`
3. `pip install -r requirements.txt`
4. Install Ollama, make sure it's running
5. `ollama pull nomic-embed-text`
6. `export OLLAMA_API_KEY=your_key`

## Notes

- Ollama runs locally for embeddings (even query text gets embedded before search) while LLM inference can be cloud-sided (`gpt-oss:20b`) so anyone can query regardless of GPU.
