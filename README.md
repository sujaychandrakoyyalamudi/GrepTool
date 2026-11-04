# LLM GrepTool

A Python file-search library and command-line tool with exact, regex, fuzzy, semantic and hybrid modes, plus LangChain tool wrappers.

## What it does

`GrepTool.search()` collects files from the paths you give it, reads each one as text, and returns a `SearchResult` for every matching line: file path, line number, line text, matched text, context lines before and after, a score and match metadata. Results are sorted by score, then by file and line, and capped at `max_results`. The same engine is available as a CLI (`python -m llm_greptool.cli`) and as LangChain `BaseTool` subclasses.

## Features

- Exact mode: substring match per line, case-insensitive by default in the library.
- Regex mode: Python `re` patterns matched per line; an invalid pattern raises `ValueError`.
- Fuzzy mode: Levenshtein distance between the query and each whole line (see limits).
- Semantic mode: embeds every line longer than 10 characters (ignoring surrounding whitespace) with `sentence-transformers` (default model `all-MiniLM-L6-v2`), ranks lines by cosine similarity, and returns the top 10 with a similarity of at least 0.3.
- Hybrid mode: exact matches plus semantic matches, merged by file and line. If semantic search fails, for example because its packages are missing, hybrid returns the exact matches only.
- Recursive directory walk that applies a simplified reading of the `.gitignore` in each directory you pass, plus `include_patterns` and `exclude_patterns` globs on file names.
- Context lines around each match (`SearchConfig.context_lines`, default 2; `-C` in the CLI).
- Compressed files: `.gz` files are decompressed and searched; `.zip` and `.tar` members are searched when `search_archives=True`.
- JSON: `SearchResult.to_dict()` and `to_json()`, and `--json` in the CLI.
- CLI output formatted with Rich: colored file headers and line numbers in bordered panels.
- LangChain: `LangChainGrepTool` (tool name `llm_grep`) and `create_grep_tools()`, which returns tools named `code_search`, `semantic_grep` and `regex_grep`.

## Install

The package is not on PyPI, and `llm_greptool/setup.py` does not build an importable package yet: it calls `find_packages()` from inside the package folder, which finds nothing. Clone the repository, install the dependencies listed in `setup.py`, and run from the repository root:

```bash
git clone https://github.com/sujaychandrakoyyalamudi/GrepTool.git
cd GrepTool
pip install "langchain>=0.1.0" "langchain-core>=0.1.0" "pydantic>=2.0.0" "rich>=13.0.0"

# Optional, for semantic and hybrid modes
pip install "sentence-transformers>=2.2.0" "scikit-learn>=1.0.0"
```

Run Python and the CLI from the `GrepTool/` directory, or add it to `PYTHONPATH`, so that `import llm_greptool` works. The `semantic` extra in `setup.py` also lists `faiss-cpu`, which the code does not use.

## Quick start

```python
from llm_greptool import GrepTool, SearchConfig, SearchMode, LangChainGrepTool

grep = GrepTool(config=SearchConfig(context_lines=1, max_results=20))
for r in grep.search(r"def _\w+_search", paths=["llm_greptool"], mode=SearchMode.REGEX):
    print(f"{r.file_path}:{r.line_number}: {r.line_content.strip()}")

# LangChain tool: pass a dict and set "mode" (it defaults to "exact")
tool = LangChainGrepTool(root_path=".")
print(tool.invoke({"query": "SearchMode", "paths": "llm_greptool", "mode": "exact", "max_results": 3}))
```

In the CLI, regex is the default mode, and matching is case-sensitive unless you pass `-i`:

```bash
python -m llm_greptool.cli "def _\w+_search" llm_greptool
python -m llm_greptool.cli -F -i "searchmode" llm_greptool               # fixed string, ignore case
python -m llm_greptool.cli -C 1 -t py --json "class SearchConfig" llm_greptool
python -m llm_greptool.cli -s "parse the gitignore file" llm_greptool    # needs the semantic packages
```

## Configuration

`SearchConfig` is a dataclass passed to `GrepTool(config=...)`. Fields the code reads:

| Field | Default | Effect |
|---|---|---|
| `mode` | `SearchMode.EXACT` | Mode used when `search()` gets no `mode` |
| `case_sensitive` | `False` | Exact and regex modes |
| `context_lines` | `2` | Lines kept before and after each match |
| `max_results` | `100` | Cap on returned results |
| `include_patterns`, `exclude_patterns` | `[]` | `fnmatch` globs on file names |
| `respect_gitignore` | `True` | Apply the `.gitignore` of each searched directory |
| `search_archives` | `False` | Search `.zip` and `.tar` members |
| `embedding_model` | `"all-MiniLM-L6-v2"` | sentence-transformers model name |
| `top_k_semantic` | `10` | Semantic results kept before the 0.3 cutoff |
| `max_edit_distance` | `2` | Fuzzy threshold multiplier |

`recursive`, `search_compressed`, `parallel` and `num_threads` also exist but are not read. The library reads no environment variables.

## Safety and limits

- The LangChain tools confine each search to `root_path`. The requested path is resolved against it with `os.path.realpath`, which follows `..` and symlinks, and a path that ends up outside the root gets an error message instead of results. Leaving out `paths` searches the whole root. Without `root_path`, paths are used as given.
- While walking a directory, the search skips symbolic links, like `grep -r`, so a link inside the root can't expose files outside it. A symlink passed directly in `paths` is followed, but the LangChain tools still reject it if it resolves outside the root.
- The LangChain tools ignore `default_mode`. The mode comes from the tool input and defaults to `"exact"`, so `code_search`, `semantic_grep` and `regex_grep` all run exact search unless the input sets `mode`.
- `.git/`, virtual environments and other hidden directories are searched unless the `.gitignore` in the searched directory excludes them. Binary files are read as UTF-8 with errors ignored.
- Only the `.gitignore` in each directory passed in `paths` is used. Negation (`!`), `**` and globs that contain `/` are not supported.
- Fuzzy mode compares the query with the whole line, so long lines rarely match, while blank and short lines match with scores near 0.
- Semantic mode embeds every eligible line on every call and keeps no index, so it is slow on large trees. The first call loads the model, which `sentence-transformers` downloads if it is not cached.
- Archive members are joined into one text: results name the archive, and line numbers count across all members. `.bz2`, `.xz` and `.tgz` are not supported.
- CLI: `-A`, `-B` and `--no-color` are accepted but ignored, and matched text is not highlighted. With `--json`, a search with no results prints a plain message and exits with code 1.

## Project layout

```text
GrepTool/
├── LICENSE
├── README.md
└── llm_greptool/
    ├── __init__.py               # exports GrepTool, SearchConfig, SearchMode, SearchResult, LangChainGrepTool
    ├── core.py                   # search engine
    ├── cli.py                    # command-line interface
    ├── langchain_integration.py  # LangChain tools and create_grep_tools()
    ├── examples.py               # usage examples
    ├── tests.py                  # pytest suite
    ├── setup.py                  # packaging metadata (does not build a usable package yet)
    └── README.md                 # detailed reference
```

## Tests

From the repository root:

```bash
pip install pytest
python -m pytest llm_greptool/tests.py -v
```

Pass the file path: pytest's default discovery looks for `test_*.py` files and skips `tests.py`. The 21 tests cover exact and regex search, context lines, the recursive walk and `.gitignore`, result serialization, the LangChain wrapper and its `root_path` confinement, including symlinks. Fuzzy, semantic, hybrid, archive search and the CLI have no tests.

## Roadmap (not implemented yet)

- AST or structural (code-aware) search
- Interactive terminal UI
- Search inside nested archives; `.bz2`, `.xz` and `.tgz`; per-member file names and line numbers
- A persistent vector index (FAISS is listed in the `setup.py` extras but not imported)
- Parallel search (`parallel`, `num_threads`), non-recursive search (`recursive`) and a switch for `.gz` reading (`search_compressed`)
- Full `.gitignore` support: negation, nested files and `**`
- Using `default_mode` in the LangChain tools
- CLI match highlighting, `-A`/`-B` and `--no-color`
- Packaging that installs with `pip`, including the `llm-grep` command declared in `setup.py`

## License

MIT. See [LICENSE](LICENSE).

## Documentation

The [detailed README](llm_greptool/README.md) covers each search mode, file selection rules, the full `SearchConfig`, the API, every CLI flag, the LangChain tools, the examples script and the tests.
