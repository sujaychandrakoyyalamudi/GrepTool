# LLM GrepTool

A powerful search tool combining traditional grep functionality with AI-powered semantic search capabilities. Designed for seamless integration with LangChain agents and chains.

## Features

### Core Search Capabilities (Must-Have)
- **Exact String Matching** - Fast fixed-string search
- **Regular Expression Search** - Full regex support with Unicode
- **Recursive Multi-file Search** - Search across directories respecting `.gitignore`
- **Context Windows** - Show surrounding lines for each match
- **Colored Output** - ANSI color highlighting for CLI

### Advanced Features (Should-Have)
- **Semantic Search** - Natural language search using embeddings
- **Fuzzy Matching** - Approximate string matching for typos
- **Hybrid Search** - Combine exact and semantic matching
- **Compressed File Support** - Search inside .gz, .zip, .tar files
- **JSON/Structured Output** - Machine-readable results for integration

### Nice-to-Have Features
- **Archive Search** - Search inside nested archives
- **AST/Structural Search** - Code-aware pattern matching
- **Interactive TUI Mode** - Terminal UI for query refinement

## Installation

```bash
# Install dependencies
pip install langchain langchain-core langchain-community sentence-transformers faiss-cpu pydantic rich

# Or install from source
cd /workspace
pip install -e .
```

## Quick Start

### Basic Usage

```python
from llm_greptool import GrepTool, SearchMode

# Create a grep tool instance
grep = GrepTool()

# Exact search
results = grep.search(
    query="DATABASE_URL",
    paths=["./src"],
    mode=SearchMode.EXACT
)

# Regex search
results = grep.search(
    query=r"def \w+\(.*password.*\)",
    paths=["."],
    mode=SearchMode.REGEX
)

# Semantic search (natural language)
results = grep.search(
    query="function that validates user email",
    paths=["./app"],
    mode=SearchMode.SEMANTIC
)
```

### LangChain Integration

```python
from llm_greptool import LangChainGrepTool
from langchain.agents import initialize_agent, AgentType
from langchain_openai import ChatOpenAI

# Create LangChain-compatible tool
grep_tool = LangChainGrepTool(
    root_path="/path/to/codebase",
    default_mode="hybrid"
)

# Use with an agent
llm = ChatOpenAI(model="gpt-4o-mini")
agent = initialize_agent(
    [grep_tool],
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
)

# Let the agent search your codebase
response = agent.run("Find all authentication-related code")
print(response)
```

### Multiple Tools for Agents

```python
from llm_greptool.langchain_integration import create_grep_tools

# Create a suite of search tools
tools = create_grep_tools(
    root_path="/workspace",
    enable_semantic=True,
    enable_regex=True
)

# tools now contains:
# - code_search: Hybrid exact+semantic search
# - semantic_grep: Natural language search
# - regex_grep: Regular expression search
```

## CLI Usage

```bash
# Exact search
python -m llm_greptool.cli "def authenticate" ./src

# Regex search
python -m llm_greptool.cli -E "pattern.*" ./code

# Semantic search
python -m llm_greptool.cli -s "user login flow" ./app

# With context lines
python -m llm_greptool.cli -C 3 "DATABASE_URL" .

# JSON output
python -m llm_greptool.cli --json "error handling" ./src
```

## API Reference

### GrepTool Class

```python
class GrepTool:
    def search(
        self,
        query: str,
        paths: Union[str, List[str]],
        mode: SearchMode = SearchMode.EXACT,
        **kwargs
    ) -> List[SearchResult]
```

**Parameters:**
- `query`: Search query (text, regex, or natural language)
- `paths`: File or directory paths to search
- `mode`: Search mode (EXACT, REGEX, FUZZY, SEMANTIC, HYBRID)
- `max_results`: Maximum number of results (default: 100)
- `context_lines`: Lines of context around matches (default: 2)
- `case_sensitive`: Case-sensitive search (default: False)

### SearchResult Class

```python
@dataclass
class SearchResult:
    file_path: str
    line_number: int
    line_content: str
    match_text: str
    context_before: List[str]
    context_after: List[str]
    score: float  # Relevance score (1.0 for exact matches)
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]
    def to_json(self) -> str
```

## Configuration

### SearchConfig

```python
from llm_greptool import SearchConfig, SearchMode

config = SearchConfig(
    mode=SearchMode.HYBRID,
    recursive=True,
    case_sensitive=False,
    context_lines=2,
    max_results=100,
    respect_gitignore=True,
    search_archives=False,
    embedding_model="all-MiniLM-L6-v2",
    top_k_semantic=10
)

grep = GrepTool(config=config)
```

## Examples

See `llm_greptool/examples.py` for comprehensive usage examples including:
- Basic search operations
- LangChain tool integration
- Semantic search with embeddings
- Multiple tools for agents
- JSON output formatting
- Context windows

## Architecture

```
┌─────────────┐
│ User/Agent  │
└──────┬──────┘
       │
       ▼
┌─────────────┐
│ CLI / API   │
└──────┬──────┘
       │
       ├──────────────┐
       ▼              ▼
┌─────────────┐ ┌──────────────┐
│ Regex Engine│ │Semantic Engine│
└──────┬──────┘ └──────┬───────┘
       │               │
       ▼               ▼
┌─────────────┐ ┌──────────────┐
│File System  │ │ Vector Index │
└─────────────┘ └──────────────┘
```

## Performance Tips

1. **Use appropriate search modes:**
   - EXACT/REGEX for known patterns (fastest)
   - SEMANTIC for conceptual searches (slower but broader)
   - HYBRID for best of both worlds

2. **Limit result sets:**
   ```python
   results = grep.search(query, paths, max_results=20)
   ```

3. **Respect gitignore:**
   ```python
   config = SearchConfig(respect_gitignore=True)
   ```

4. **Use file type filters:**
   ```python
   config = SearchConfig(include_patterns=['*.py', '*.js'])
   ```

## License

MIT License

## Contributing

Contributions welcome! Please read our contributing guidelines before submitting PRs.
