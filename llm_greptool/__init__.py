"""
LLM GrepTool - A powerful search tool combining traditional grep with AI-powered semantic search.

This tool integrates with LangChain to provide both exact/regex matching and semantic search
capabilities for codebases and text files.
"""

__version__ = "0.1.0"
__author__ = "LLM GrepTool Team"

from .core import GrepTool, SearchMode, SearchResult, SearchConfig
from .langchain_integration import LLMGrepTool as LangChainGrepTool

__all__ = [
    "GrepTool",
    "SearchMode",
    "SearchResult",
    "SearchConfig",
    "LangChainGrepTool",
]
