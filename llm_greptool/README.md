# LLM GrepTool reference

Detailed reference for the `llm_greptool` package. For an overview, see the [repository README](../README.md).

## Install

The package is not on PyPI. `setup.py` sits inside the package folder and calls `find_packages()`, which finds no packages there, so `pip install` from this folder (or from Git with `#subdirectory=llm_greptool`) does not install an importable `llm_greptool`, and the `llm-grep` command it declares fails. Until that is fixed, clone the repository, install the dependencies listed in `setup.py`, and run from the repository root:

```bash
git clone https://github.com/sujaychandrakoyyalamudi/GrepTool.git
cd GrepTool
pip install "langchain>=0.1.0" "langchain-core>=0.1.0" "pydantic>=2.0.0" "rich>=13.0.0"

# Optional, for semantic and hybrid modes
pip install "sentence-transformers>=2.2.0" "scikit-learn>=1.0.0"
```

Run Python and the CLI from the `GrepTool/` directory, or add it to `PYTHONPATH`. Importing the package also imports `langchain_integration`, so `langchain-core` and `pydantic` are needed even for plain searches; the CLI also needs `rich`. Only `examples.py` uses `langchain` itself. The `semantic` extra in `setup.py` also lists `faiss-cpu`, which the code does not import.

## Quick start

```python
from llm_greptool import GrepTool, SearchConfig, SearchMode

grep = GrepTool()

# Exact substring (case-insensitive by default)
results = grep.search("DATABASE_URL", paths=["."], mode=SearchMode.EXACT)

# Regex
results = grep.search(r"def \w+\(.*password.*\)", paths=["."], mode=SearchMode.REGEX)

# Semantic (needs sentence-transformers and scikit-learn)
results = grep.search("function that validates user email", paths=["."], mode=SearchMode.SEMANTIC)

for r in results:
    print(f"{r.file_path}:{r.line_number} ({r.score:.2f}) {r.line_content.strip()}")

# Configured instance
config = SearchConfig(
    mode=SearchMode.HYBRID,
    context_lines=3,
    max_results=50,
    include_patterns=["*.py"],
    respect_gitignore=True,
)
hybrid = GrepTool(config=config)
results = hybrid.search("load configuration", paths=["llm_greptool"])
```

## Search modes

`SearchMode` is an `Enum`. `GrepTool.search()` needs a member such as `SearchMode.REGEX`; a plain string raises `ValueError("Unknown search mode: ...")`.

| Mode | Behavior |
|---|---|
| `EXACT` | A line matches if it contains the query. Case-insensitive unless `case_sensitive=True`. `score` is 1.0 and `match_text` is the query. |
| `REGEX` | `re.compile(query)`, with `re.IGNORECASE` unless case-sensitive, and `search()` on each line. `match_text` is the first match on the line. An invalid pattern raises `ValueError("Invalid regex pattern: ...")`. |
| `FUZZY` | Pure-Python Levenshtein distance between the query and the whole line. A line matches when the distance is at most `max_edit_distance * len(query)`. `score` is `1 - distance / max(len(query), len(line))`, floored at 0. Case is not normalized. |
| `SEMANTIC` | Loads `SentenceTransformer(embedding_model)` once per `GrepTool` instance. On every call it embeds each line whose stripped text is longer than 10 characters, computes cosine similarity to the query with scikit-learn, keeps the `top_k_semantic` best lines, and drops those below 0.3. `score` is the similarity. Raises `ImportError` if either package is missing. |
| `HYBRID` | Exact results plus semantic results, de-duplicated by file and line with exact results kept first. Exact matches score 1.0. Any exception from the semantic step is swallowed, leaving exact results only. |

Each mode produces one result per matching line, not one per occurrence. `search()` sorts results by score (descending), then file path and line number, and returns the first `max_results`.

Fuzzy matching compares whole lines, not substrings. A long line that contains the query rarely matches, while blank and short lines match with scores near 0. For example, with the default `max_edit_distance=2`, every blank line matches with a score of 0.

## File selection

- `paths` may be a string or a list. Files are searched directly. Directories are walked with `os.walk`, always recursively. Paths that do not exist are skipped without an error.
- When `respect_gitignore` is on, the walk reads `.gitignore` from each directory passed in `paths`. Parent directories and nested `.gitignore` files are not read. Pattern handling is simplified:
  - `#` comments and blank lines are skipped.
  - `/name` ignores any path whose text ends with `name`.
  - `name/` ignores directories called `name`.
  - Patterns containing `*` are matched with `fnmatch` against the file or directory name only.
  - Other patterns ignore entries with that exact name, or paths ending in `/pattern`.
  - Negation (`!`) and `**` are not supported.
- `exclude_patterns` and `include_patterns` are `fnmatch` globs on the file name. They apply to files found while walking directories, not to files passed directly. Exclusion wins; a non-empty `include_patterns` keeps only matching names.
- Nothing else is skipped. `.git/`, virtual environments and other hidden directories are walked unless the `.gitignore` excludes them, and binary files are read as UTF-8 with errors ignored.
- Files are read by suffix:
  - `.gz`: decompressed with `gzip` and searched as text, always. A `.tar.gz` file is handled the same way, so its raw tar stream is searched as text.
  - `.zip` and `.tar`: searched only when `search_archives=True`. The text of all members is joined with newlines, so results show the archive path and line numbers count lines across all members.
  - `.bz2` and `.xz`: skipped, with or without `search_archives`.
  - Everything else, including `.tgz`: read as UTF-8 text with errors ignored.
  - Files that cannot be read are skipped.

## Configuration

`SearchConfig` is a dataclass. The library reads no environment variables.

| Field | Default | Used for |
|---|---|---|
| `mode` | `SearchMode.EXACT` | Mode when `search()` gets no `mode` argument |
| `case_sensitive` | `False` | Exact and regex modes |
| `context_lines` | `2` | Lines before and after each match, in all modes |
| `max_results` | `100` | Maximum results returned by `search()` |
| `include_patterns` | `[]` | File-name globs to keep |
| `exclude_patterns` | `[]` | File-name globs to skip |
| `respect_gitignore` | `True` | Read the `.gitignore` in each searched directory |
| `search_archives` | `False` | Search `.zip` and `.tar` members |
| `embedding_model` | `"all-MiniLM-L6-v2"` | sentence-transformers model; read when the model is first loaded |
| `top_k_semantic` | `10` | Semantic results kept before the 0.3 cutoff |
| `max_edit_distance` | `2` | Fuzzy threshold multiplier |
| `recursive` | `True` | Not used; directory search is always recursive |
| `search_compressed` | `True` | Not used; `.gz` files are always decompressed |
| `parallel` | `True` | Not used; files are searched one at a time |
| `num_threads` | `4` | Not used |

## API reference

### `GrepTool(config: Optional[SearchConfig] = None)`

#### `search(query, paths, mode=None, **kwargs) -> List[SearchResult]`

- `query`: text, regex pattern or natural-language query, depending on the mode.
- `paths`: `str` or `List[str]`.
- `mode`: a `SearchMode` member; `None` uses `config.mode`.
- Keyword arguments read per call: `max_results`, `case_sensitive` (exact, regex and the exact part of hybrid), `max_edit_distance` (fuzzy), `top_k` (semantic) and `embedding_model` (semantic, only before the model is first loaded). Other keyword arguments are ignored, including `context_lines`; set context lines in `SearchConfig`.

#### `search_generator(query, paths, mode=None, **kwargs) -> Iterator[SearchResult]`

Yields results for `EXACT` and `REGEX` only; other modes yield nothing. It computes all results for the mode before yielding the first one, and it neither sorts nor applies `max_results`.

### `SearchResult`

```python
@dataclass
class SearchResult:
    file_path: str
    line_number: int                                         # 1-based
    line_content: str
    match_text: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    score: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)  # includes "match_type"

    def to_dict(self) -> Dict[str, Any]: ...
    def to_json(self) -> str: ...                            # indented JSON
```

`metadata["match_type"]` is `exact`, `regex`, `fuzzy` or `semantic`. Regex results also carry `metadata["pattern"]`, fuzzy results `metadata["edit_distance"]`, and semantic results `metadata["embedding_model"]`.

## CLI

```bash
python -m llm_greptool.cli [options] QUERY PATH [PATH ...]
```

At least one path is required. The help text mentions a default of the current directory, but argparse is configured with `nargs="+"`, so a path must be given.

| Option | Effect |
|---|---|
| none, or `-E`, `--extended-regex` | Regex mode (the CLI default) |
| `-F`, `--fixed-strings` | Exact mode |
| `-z`, `--fuzzy` | Fuzzy mode |
| `-s`, `--semantic` | Semantic mode |
| `--hybrid` | Hybrid mode |
| `-i`, `--ignore-case` | Case-insensitive matching; the CLI is case-sensitive by default |
| `-C NUM`, `--context NUM` | Context lines (default 2) |
| `-l`, `--files-with-matches` | Print only the paths of matching files |
| `-t TYPE`, `--type TYPE` | Only files named `*.TYPE`; can be repeated |
| `--no-gitignore` | Do not read `.gitignore` |
| `--max-results N` | Maximum results (default 100) |
| `--embedding-model NAME` | Semantic model (default `all-MiniLM-L6-v2`) |
| `--top-k N` | Semantic results to keep (default 10) |
| `--json` | Print `{"query", "count", "results"}`, where each result is `SearchResult.to_dict()` |
| `-V`, `--version` | Print `llm-grep 0.1.0` |
| `-A NUM`, `-B NUM`, `--no-color` | Accepted, no effect |
| `-r`, `--recursive`, `-n`, `--line-numbers` | Always on; the flags change nothing |

Human-readable output shows each match in a Rich panel with the file path, line numbers and context lines. The matched text is not highlighted: `highlight_match()` builds a styled `Text`, but `format_result_human()` uses its plain string.

Exit codes: 0 when there are results, 1 when there are none (a plain "No results found" message is printed, even with `--json`), 2 on errors such as an invalid regex, and 130 on Ctrl+C.

Examples, run from the repository root:

```bash
python -m llm_greptool.cli -F "def authenticate" ./src     # exact
python -m llm_greptool.cli "pattern.*" ./code              # regex
python -m llm_greptool.cli -s "user login flow" ./app      # semantic
python -m llm_greptool.cli -C 3 "DATABASE_URL" .           # 3 context lines
python -m llm_greptool.cli --json "error handling" ./src   # JSON output
```

## LangChain integration

### `LangChainGrepTool`

`LangChainGrepTool` is `llm_greptool.langchain_integration.LLMGrepTool`, a `langchain_core.tools.BaseTool` subclass named `llm_grep`.

```python
LangChainGrepTool(root_path=None, default_mode="exact", config=None, **kwargs)
```

Extra keyword arguments, such as `name` and `description`, go to `BaseTool`. The input schema, `GrepToolInput`, has these fields:

| Field | Default | Notes |
|---|---|---|
| `query` | required | |
| `paths` | required | A single path string |
| `mode` | `"exact"` | `exact`, `regex`, `fuzzy`, `semantic` or `hybrid`; unknown values fall back to exact |
| `case_sensitive` | `False` | |
| `max_results` | `20` | |
| `context_lines` | `2` | |

For each call the tool sets `context_lines`, `max_results` and `case_sensitive` on its config from the input, runs the search, restores the old values, and returns text listing each match with path, line, score, match type and context lines. With no matches it returns `No results found for query: '...' (mode: ...)`.

```python
from llm_greptool import LangChainGrepTool

tool = LangChainGrepTool(root_path=".")
print(tool.invoke({"query": "def _get_context", "paths": "llm_greptool", "mode": "exact"}))
```

Because the input has several fields, the tool needs an agent that supports tools with more than one argument.

Behavior to know about:

- `root_path` is prepended to `paths` unless `paths` starts with `/`. Absolute paths and `..` are not blocked, so `root_path` does not limit what the tool can read.
- `default_mode` is stored but not used. The mode always comes from the input, which defaults to `"exact"`.
- `_arun` calls the synchronous `_run`.

### `create_grep_tools()`

```python
from llm_greptool.langchain_integration import create_grep_tools

tools = create_grep_tools(
    root_path=".",
    enable_semantic=True,
    enable_regex=True,
    embedding_model="all-MiniLM-L6-v2",
)
```

It returns up to three tools:

| Name | Class | Config |
|---|---|---|
| `code_search` | `LLMGrepTool`, `default_mode="hybrid"` | `max_results=50`, `context_lines=2` |
| `semantic_grep` (if `enable_semantic`) | `SemanticGrepTool` | `mode=SEMANTIC`, `top_k_semantic=15` |
| `regex_grep` (if `enable_regex`) | `RegexGrepTool` | `mode=REGEX` |

Because the mode comes from the tool input, all three tools run exact search unless the caller passes `mode`. The `semantic_grep` and `regex_grep` descriptions do not ask the model to send `mode`. The input's `max_results` (default 20) and `context_lines` (default 2) also replace the config values on every call.

## Examples

`llm_greptool/examples.py` has seven functions: basic search, the LangChain tool, semantic search, `create_grep_tools()`, an agent, JSON output and context windows. Run them from the repository root:

```bash
python -m llm_greptool.examples
```

- `main()` runs the examples in order without error handling, so the semantic example stops the script if `sentence-transformers` or `scikit-learn` is missing.
- The agent example runs only when `initialize_agent` can be imported from `langchain.agents` (a LangChain 0.x API), `langchain-openai` is installed and `OPENAI_API_KEY` is set; otherwise it prints a skip message. It builds a `ZERO_SHOT_REACT_DESCRIPTION` agent, and its errors are caught and printed.
- The LangChain examples use `root_path="/workspace"`, which may not exist on your machine.

## Tests

From the repository root:

```bash
pip install pytest
python -m pytest llm_greptool/tests.py -v
```

Pass the file path: pytest's default discovery looks for `test_*.py` files and skips `tests.py`. The test module imports `llm_greptool` as a package, so it needs `langchain-core` and `pydantic`.

The 14 tests cover exact search and case-insensitivity, regex search and invalid patterns, context lines, the recursive walk and `.gitignore`, `to_dict()` and `to_json()`, and creating and running `LLMGrepTool`. Fuzzy, semantic, hybrid, archive search and the CLI have no tests.

## Roadmap (not implemented yet)

- AST or structural (code-aware) search
- Interactive terminal UI
- Search inside nested archives, `.bz2`, `.xz` and `.tgz` support, and per-member file names and line numbers
- A persistent vector index (FAISS is listed in the `setup.py` extras but not imported); embeddings are recomputed on every call today
- Parallel search (`parallel`, `num_threads`)
- Non-recursive search (`recursive`) and a switch for `.gz` reading (`search_compressed`)
- Full `.gitignore` support: negation, nested files and `**`
- Using `default_mode` in the LangChain tools
- `search_generator()` support for fuzzy, semantic and hybrid modes
- CLI match highlighting, `-A`/`-B` and `--no-color`
- Packaging that installs with `pip`, including the `llm-grep` command

## License

MIT. See [LICENSE](../LICENSE).
