#!/usr/bin/env python3
"""
CLI interface for LLM GrepTool.

Provides a command-line interface similar to traditional grep but with
AI-powered semantic search capabilities.
"""

import sys
import argparse
import json
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel
from rich.text import Text

from .core import GrepTool, SearchMode, SearchResult, SearchConfig


console = Console()


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="llm-grep",
        description="LLM-powered grep tool with exact, regex, and semantic search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  llm-grep "def authenticate" ./src                    # Exact search
  llm-grep -r "pattern.*" ./code                       # Regex search
  llm-grep -s "user login flow" ./app                  # Semantic search
  llm-grep -C 3 "DATABASE_URL" .                       # With context
  llm-grep --json "error handling" ./src               # JSON output
        """
    )
    
    # Query and paths
    parser.add_argument(
        "query",
        help="Search query (text, regex pattern, or natural language)"
    )
    parser.add_argument(
        "paths",
        nargs="+",
        default=["."],
        help="File or directory paths to search (default: current directory)"
    )
    
    # Search mode
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "-F", "--fixed-strings",
        action="store_true",
        help="Treat query as fixed string (not regex)"
    )
    mode_group.add_argument(
        "-E", "--extended-regex",
        action="store_true",
        help="Use extended regex (default)"
    )
    mode_group.add_argument(
        "-s", "--semantic",
        action="store_true",
        help="Use semantic search (natural language)"
    )
    mode_group.add_argument(
        "-z", "--fuzzy",
        action="store_true",
        help="Use fuzzy matching"
    )
    mode_group.add_argument(
        "--hybrid",
        action="store_true",
        help="Use hybrid search (exact + semantic)"
    )
    
    # Context options
    parser.add_argument(
        "-C", "--context",
        type=int,
        default=2,
        metavar="NUM",
        help="Number of context lines around matches (default: 2)"
    )
    parser.add_argument(
        "-B", "--before-context",
        type=int,
        default=None,
        metavar="NUM",
        help="Lines of context before matches"
    )
    parser.add_argument(
        "-A", "--after-context",
        type=int,
        default=None,
        metavar="NUM",
        help="Lines of context after matches"
    )
    
    # Output options
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results in JSON format"
    )
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable colored output"
    )
    parser.add_argument(
        "-n", "--line-numbers",
        action="store_true",
        default=True,
        help="Show line numbers (default: true)"
    )
    parser.add_argument(
        "-l", "--files-with-matches",
        action="store_true",
        help="Only show filenames with matches"
    )
    
    # Filtering options
    parser.add_argument(
        "-i", "--ignore-case",
        action="store_true",
        help="Case-insensitive search"
    )
    parser.add_argument(
        "-r", "--recursive",
        action="store_true",
        default=True,
        help="Recursive search (default: true)"
    )
    parser.add_argument(
        "--no-gitignore",
        action="store_true",
        help="Don't respect .gitignore files"
    )
    parser.add_argument(
        "-t", "--type",
        action="append",
        metavar="TYPE",
        help="Filter by file type (e.g., py, js, ts)"
    )
    parser.add_argument(
        "--max-results",
        type=int,
        default=100,
        help="Maximum number of results (default: 100)"
    )
    
    # Semantic search options
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="all-MiniLM-L6-v2",
        help="Model for semantic embeddings"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Top K results for semantic search"
    )
    
    # Version and help
    parser.add_argument(
        "-V", "--version",
        action="version",
        version="llm-grep 0.1.0"
    )
    
    return parser.parse_args(argv)


def determine_search_mode(args: argparse.Namespace) -> SearchMode:
    """Determine search mode from arguments."""
    if args.semantic:
        return SearchMode.SEMANTIC
    elif args.fuzzy:
        return SearchMode.FUZZY
    elif args.hybrid:
        return SearchMode.HYBRID
    elif args.fixed_strings:
        return SearchMode.EXACT
    else:
        return SearchMode.REGEX


def highlight_match(line: str, match_text: str, case_sensitive: bool = False) -> Text:
    """Create highlighted text for a match."""
    text = Text(line)
    
    search_text = match_text if case_sensitive else match_text.lower()
    line_text = line if case_sensitive else line.lower()
    
    start = 0
    while True:
        idx = line_text.find(search_text, start)
        if idx == -1:
            break
        text.stylize("bold red", idx, idx + len(match_text))
        start = idx + 1
    
    return text


def format_result_human(
    result: SearchResult,
    show_line_numbers: bool = True,
    context_before: Optional[int] = None,
    context_after: Optional[int] = None
) -> str:
    """Format a single result for human-readable output."""
    lines = []
    
    # File header
    file_line = f"{result.file_path}:{result.line_number}"
    lines.append(f"[bold blue]{file_line}[/bold blue]")
    
    # Context before
    if result.context_before:
        for i, ctx_line in enumerate(result.context_before):
            line_num = result.line_number - len(result.context_before) + i
            lines.append(f"  [dim]{line_num:4d}│[/dim] {ctx_line}")
    
    # Match line with highlighting
    highlighted = highlight_match(result.line_content, result.match_text)
    line_prefix = f"→ {result.line_number:4d}│" if show_line_numbers else "→ "
    lines.append(f"  [bold]{line_prefix}[/bold] {highlighted.plain}")
    
    # Context after
    if result.context_after:
        for i, ctx_line in enumerate(result.context_after):
            line_num = result.line_number + i + 1
            lines.append(f"  [dim]{line_num:4d}│[/dim] {ctx_line}")
    
    return "\n".join(lines)


def print_results(
    results: List[SearchResult],
    args: argparse.Namespace,
    query: str
) -> None:
    """Print search results in appropriate format."""
    if not results:
        console.print(f"[yellow]No results found for:[/yellow] '{query}'")
        sys.exit(1)
    
    if args.json:
        # JSON output
        output = {
            "query": query,
            "count": len(results),
            "results": [r.to_dict() for r in results]
        }
        print(json.dumps(output, indent=2))
    elif args.files_with_matches:
        # Only filenames
        seen_files = set()
        for result in results:
            if result.file_path not in seen_files:
                console.print(result.file_path)
                seen_files.add(result.file_path)
    else:
        # Human-readable output with Rich formatting
        console.print(f"\n[bold green]Found {len(results)} result(s) for:[/bold green] '{query}'\n")
        console.print("=" * 80)
        
        for i, result in enumerate(results, 1):
            if i > args.max_results:
                break
            
            console.print(
                Panel(
                    format_result_human(
                        result,
                        show_line_numbers=args.line_numbers,
                        context_before=args.before_context or args.context,
                        context_after=args.after_context or args.context
                    ),
                    title=f"[bold]Match {i}[/bold]",
                    border_style="blue",
                    padding=(0, 1)
                )
            )
            console.print()
        
        if len(results) > args.max_results:
            console.print(f"\n[yellow]Showing {args.max_results} of {len(results)} results[/yellow]")


def main(argv: Optional[List[str]] = None) -> int:
    """Main entry point for CLI."""
    args = parse_args(argv)
    
    # Determine search mode
    mode = determine_search_mode(args)
    
    # Create configuration
    config = SearchConfig(
        mode=mode,
        recursive=args.recursive,
        case_sensitive=not args.ignore_case,
        context_lines=args.context,
        max_results=args.max_results,
        respect_gitignore=not args.no_gitignore,
        embedding_model=args.embedding_model,
        top_k_semantic=args.top_k,
    )
    
    # Set file type filters
    if args.type:
        include_patterns = []
        for t in args.type:
            if not t.startswith('.'):
                t = '.' + t
            include_patterns.append(f'*{t}')
        config.include_patterns = include_patterns
    
    # Create grep tool
    grep_tool = GrepTool(config=config)
    
    try:
        # Perform search
        results = grep_tool.search(
            query=args.query,
            paths=args.paths,
            mode=mode,
        )
        
        # Print results
        print_results(results, args, args.query)
        
        return 0 if results else 1
        
    except KeyboardInterrupt:
        console.print("\n[yellow]Search interrupted[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]Error:[/red] {e}")
        return 2


if __name__ == "__main__":
    sys.exit(main())
