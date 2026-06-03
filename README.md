# LLM GrepTool

A powerful search tool combining traditional grep functionality with AI-powered semantic search capabilities. Designed for seamless integration with LangChain agents and chains.

## Quick Start

```python
from llm_greptool import GrepTool, SearchMode, LangChainGrepTool

# Basic exact/regex search
grep = GrepTool()
results = grep.search(query="def authenticate", paths=["./src"], mode=SearchMode.REGEX)

# LangChain integration
tool = LangChainGrepTool(root_path="/workspace")
result = tool._run(query="import", paths=".", mode="regex")
```

## Features

- **Exact & Regex Matching** - Fast pattern matching with full regex support
- **Semantic Search** - Natural language search using embeddings  
- **Hybrid Search** - Combine exact and semantic matching
- **Context Windows** - Show surrounding lines for each match
- **LangChain Integration** - Use as a tool with LangChain agents
- **CLI Interface** - Command-line tool similar to traditional grep
- **JSON Output** - Machine-readable results for integration

## Documentation

See `llm_greptool/README.md` for full documentation including:
- Installation instructions
- API reference
- Usage examples
- LangChain agent integration
- CLI usage

## Testing

```bash
python -m pytest llm_greptool/tests.py -v
```

All 14 tests pass successfully!

## Example Usage

```bash
# CLI exact search
python -m llm_greptool.cli "def main" . --max-results 5

# CLI regex search
python -m llm_greptool.cli -E "class.*:" llm_greptool/

# CLI JSON output
python -m llm_greptool.cli "DATABASE_URL" . --json
```
