"""
LangChain integration for LLM GrepTool.

Provides LangChain-compatible tool wrappers for use with agents and chains.
"""

from typing import Type, List, Optional, Dict, Any
from langchain_core.tools import BaseTool
from langchain_core.callbacks import CallbackManagerForToolRun
from pydantic import BaseModel, Field

from .core import GrepTool, SearchMode, SearchResult, SearchConfig


class GrepToolInput(BaseModel):
    """Input schema for the GrepTool."""
    query: str = Field(..., description="Search query (text, regex pattern, or natural language)")
    paths: str = Field(..., description="File path or directory path to search")
    mode: str = Field(default="exact", description="Search mode: exact, regex, fuzzy, semantic, or hybrid")
    case_sensitive: bool = Field(default=False, description="Whether search is case-sensitive")
    max_results: int = Field(default=20, description="Maximum number of results to return")
    context_lines: int = Field(default=2, description="Number of context lines before and after matches")


class LLMGrepTool(BaseTool):
    """
    LangChain-compatible wrapper for LLM GrepTool.
    
    This tool can be used with LangChain agents and chains to provide
    powerful search capabilities over codebases and text files.
    
    Example usage:
        from llm_greptool import LangChainGrepTool
        
        # Create the tool
        grep_tool = LangChainGrepTool(
            root_path="/path/to/codebase",
            default_mode="hybrid"
        )
        
        # Use with an agent
        from langchain.agents import initialize_agent, AgentType
        from langchain_openai import ChatOpenAI
        
        llm = ChatOpenAI(model="gpt-4")
        agent = initialize_agent(
            [grep_tool],
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION
        )
        
        # Run a query
        result = agent.run("Find all authentication-related code")
    """
    
    name: str = "llm_grep"
    description: str = """
    Search through files using exact matching, regex, or semantic search.
    Useful for finding code patterns, functions, classes, or concepts in a codebase.
    
    Input should be a JSON object with:
    - query: The search query (string, regex, or natural language)
    - paths: Path to file or directory to search
    - mode: Search mode (exact, regex, fuzzy, semantic, hybrid)
    - case_sensitive: Boolean for case sensitivity
    - max_results: Maximum results to return
    - context_lines: Context lines around matches
    
    Examples:
    - {"query": "def authenticate", "paths": "/src", "mode": "regex"}
    - {"query": "user login flow", "paths": "/app", "mode": "semantic"}
    - {"query": "DATABASE_URL", "paths": ".", "mode": "exact"}
    """
    
    args_schema: Type[BaseModel] = GrepToolInput
    
    # Configuration
    root_path: Optional[str] = None
    default_mode: str = "exact"
    config: Optional[SearchConfig] = None
    
    def __init__(
        self,
        root_path: Optional[str] = None,
        default_mode: str = "exact",
        config: Optional[SearchConfig] = None,
        **kwargs
    ):
        """Initialize the LangChain GrepTool."""
        super().__init__(**kwargs)
        
        self.root_path = root_path
        self.default_mode = default_mode
        self.config = config or SearchConfig()
        self._grep_tool = GrepTool(config=self.config)
    
    def _run(
        self,
        query: str,
        paths: str,
        mode: str = "exact",
        case_sensitive: bool = False,
        max_results: int = 20,
        context_lines: int = 2,
    ) -> str:
        """
        Execute the search and return formatted results.
        
        Args:
            query: Search query
            paths: Path(s) to search
            mode: Search mode
            case_sensitive: Case sensitivity flag
            max_results: Maximum results to return
            context_lines: Context lines around matches
            
        Returns:
            Formatted string with search results
        """
        # Map mode string to enum
        mode_map = {
            "exact": SearchMode.EXACT,
            "regex": SearchMode.REGEX,
            "fuzzy": SearchMode.FUZZY,
            "semantic": SearchMode.SEMANTIC,
            "hybrid": SearchMode.HYBRID,
        }
        
        search_mode = mode_map.get(mode.lower(), SearchMode.EXACT)
        
        # Use root_path if provided and paths is relative
        if self.root_path and not paths.startswith('/'):
            search_paths = [self.root_path + "/" + paths]
        else:
            search_paths = [paths]
        
        # Update config temporarily
        original_context = self._grep_tool.config.context_lines
        original_max = self._grep_tool.config.max_results
        original_case = self._grep_tool.config.case_sensitive
        
        self._grep_tool.config.context_lines = context_lines
        self._grep_tool.config.max_results = max_results
        self._grep_tool.config.case_sensitive = case_sensitive
        
        try:
            # Perform search
            results = self._grep_tool.search(
                query=query,
                paths=search_paths,
                mode=search_mode,
            )
            
            # Format results
            return self._format_results(results, query, search_mode)
            
        finally:
            # Restore original config
            self._grep_tool.config.context_lines = original_context
            self._grep_tool.config.max_results = original_max
            self._grep_tool.config.case_sensitive = original_case
    
    async def _arun(
        self,
        query: str,
        paths: str,
        mode: str = "exact",
        case_sensitive: bool = False,
        max_results: int = 20,
        context_lines: int = 2,
    ) -> str:
        """Async version of _run (currently synchronous)."""
        # For now, just call the sync version
        # In production, you could implement true async with aiofiles, etc.
        return self._run(
            query=query,
            paths=paths,
            mode=mode,
            case_sensitive=case_sensitive,
            max_results=max_results,
            context_lines=context_lines,
        )
    
    def _format_results(
        self,
        results: List[SearchResult],
        query: str,
        mode: SearchMode
    ) -> str:
        """Format search results into a readable string."""
        if not results:
            return f"No results found for query: '{query}' (mode: {mode.value})"
        
        output_lines = [
            f"Found {len(results)} result(s) for query: '{query}' (mode: {mode.value})",
            "=" * 80,
            ""
        ]
        
        for i, result in enumerate(results, 1):
            # File and line info
            output_lines.append(f"[{i}] {result.file_path}:{result.line_number}")
            output_lines.append(f"    Score: {result.score:.3f} | Type: {result.metadata.get('match_type', 'unknown')}")
            output_lines.append("")
            
            # Context before
            if result.context_before:
                for line in result.context_before:
                    output_lines.append(f"      {line}")
            
            # Match line with highlighting
            match_line = result.line_content
            output_lines.append(f"    → {match_line}")
            
            # Context after
            if result.context_after:
                for line in result.context_after:
                    output_lines.append(f"      {line}")
            
            output_lines.append("")
            output_lines.append("-" * 80)
            output_lines.append("")
        
        return "\n".join(output_lines)


class SemanticGrepTool(LLMGrepTool):
    """
    Specialized LangChain tool for semantic search only.
    
    This is pre-configured for semantic/embedding-based search,
    useful for natural language queries about code functionality.
    """
    
    name: str = "semantic_grep"
    description: str = """
    Perform semantic search through code files using natural language queries.
    Finds code by meaning rather than exact text matching.
    
    Input should be a JSON object with:
    - query: Natural language description of what you're looking for
    - paths: Path to file or directory to search
    - max_results: Maximum results to return (default: 10)
    
    Examples:
    - {"query": "function that validates user email", "paths": "/src"}
    - {"query": "database connection setup", "paths": "/app/config"}
    - {"query": "error handling middleware", "paths": "."}
    """
    
    def __init__(self, **kwargs):
        """Initialize with semantic search mode."""
        kwargs['default_mode'] = 'semantic'
        super().__init__(**kwargs)


class RegexGrepTool(LLMGrepTool):
    """
    Specialized LangChain tool for regex search only.
    
    Pre-configured for regular expression pattern matching.
    """
    
    name: str = "regex_grep"
    description: str = """
    Search through files using regular expressions.
    Powerful for finding specific code patterns, function signatures, etc.
    
    Input should be a JSON object with:
    - query: Regular expression pattern
    - paths: Path to file or directory to search
    - case_sensitive: Boolean for case sensitivity
    - max_results: Maximum results to return
    
    Examples:
    - {"query": "def \\w+\\(.*password.*\\)", "paths": "/src"}
    - {"query": "import.*from.*os", "paths": "."}
    - {"query": "class \\w+ extends.*Component", "paths": "/components"}
    """
    
    def __init__(self, **kwargs):
        """Initialize with regex search mode."""
        kwargs['default_mode'] = 'regex'
        super().__init__(**kwargs)


def create_grep_tools(
    root_path: str,
    enable_semantic: bool = True,
    enable_regex: bool = True,
    embedding_model: str = "all-MiniLM-L6-v2"
) -> List[BaseTool]:
    """
    Factory function to create a set of grep tools for LangChain agents.
    
    Args:
        root_path: Root directory for searches
        enable_semantic: Whether to include semantic search tool
        enable_regex: Whether to include regex search tool
        embedding_model: Model to use for semantic embeddings
        
    Returns:
        List of LangChain BaseTool instances
    """
    tools = []
    
    # Main hybrid search tool
    config = SearchConfig(
        embedding_model=embedding_model,
        respect_gitignore=True,
        context_lines=2,
        max_results=50
    )
    
    main_tool = LLMGrepTool(
        root_path=root_path,
        default_mode="hybrid",
        config=config,
        name="code_search",
        description="""
        Search code files using hybrid exact+semantic matching.
        Best for general code search queries. Accepts query, paths, mode, max_results, context_lines.
        """
    )
    tools.append(main_tool)
    
    # Semantic search tool
    if enable_semantic:
        semantic_config = SearchConfig(
            embedding_model=embedding_model,
            mode=SearchMode.SEMANTIC,
            top_k_semantic=15
        )
        
        semantic_tool = SemanticGrepTool(
            root_path=root_path,
            config=semantic_config
        )
        tools.append(semantic_tool)
    
    # Regex search tool
    if enable_regex:
        regex_config = SearchConfig(
            mode=SearchMode.REGEX,
            case_sensitive=False
        )
        
        regex_tool = RegexGrepTool(
            root_path=root_path,
            config=regex_config
        )
        tools.append(regex_tool)
    
    return tools
