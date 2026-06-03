"""Tests for LLM GrepTool."""

import pytest
import tempfile
import os
from pathlib import Path

from llm_greptool import GrepTool, SearchMode, SearchResult, SearchConfig
from llm_greptool.langchain_integration import LLMGrepTool


class TestExactSearch:
    """Test exact string matching."""
    
    def test_exact_match_found(self, tmp_path):
        """Test that exact matches are found."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nfoo bar\nhello again\n")
        
        grep = GrepTool()
        results = grep.search(
            query="hello",
            paths=[str(tmp_path)],
            mode=SearchMode.EXACT
        )
        
        assert len(results) == 2
        assert all(r.match_text == "hello" for r in results)
    
    def test_exact_match_not_found(self, tmp_path):
        """Test that non-existent patterns return empty results."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nfoo bar\n")
        
        grep = GrepTool()
        results = grep.search(
            query="notfound",
            paths=[str(tmp_path)],
            mode=SearchMode.EXACT
        )
        
        assert len(results) == 0
    
    def test_case_insensitive(self, tmp_path):
        """Test case-insensitive search."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World\nHELLO AGAIN\nhello there\n")
        
        grep = GrepTool(config=SearchConfig(case_sensitive=False))
        results = grep.search(
            query="hello",
            paths=[str(tmp_path)],
            mode=SearchMode.EXACT
        )
        
        assert len(results) == 3


class TestRegexSearch:
    """Test regular expression search."""
    
    def test_regex_match(self, tmp_path):
        """Test regex pattern matching."""
        test_file = tmp_path / "test.py"
        test_file.write_text("def foo():\n    pass\n\ndef bar():\n    pass\n")
        
        grep = GrepTool()
        results = grep.search(
            query=r"def \w+\(\):",
            paths=[str(tmp_path)],
            mode=SearchMode.REGEX
        )
        
        assert len(results) == 2
        assert all(r.metadata.get('match_type') == 'regex' for r in results)
    
    def test_invalid_regex(self, tmp_path):
        """Test that invalid regex raises error."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("some text\n")
        
        grep = GrepTool()
        
        with pytest.raises(ValueError, match="Invalid regex"):
            grep.search(
                query=r"[invalid(regex",
                paths=[str(tmp_path)],
                mode=SearchMode.REGEX
            )


class TestContextWindows:
    """Test context line functionality."""
    
    def test_context_lines(self, tmp_path):
        """Test that context lines are included."""
        test_file = tmp_path / "test.txt"
        content = "\n".join([f"line {i}" for i in range(1, 11)])
        test_file.write_text(content)
        
        config = SearchConfig(context_lines=2)
        grep = GrepTool(config=config)
        
        results = grep.search(
            query="line 5",
            paths=[str(tmp_path)],
            mode=SearchMode.EXACT
        )
        
        assert len(results) == 1
        result = results[0]
        
        assert len(result.context_before) == 2
        assert len(result.context_after) == 2
        assert "line 3" in result.context_before
        assert "line 4" in result.context_before
        assert "line 6" in result.context_after
        assert "line 7" in result.context_after


class TestFileCollection:
    """Test file collection and filtering."""
    
    def test_recursive_search(self, tmp_path):
        """Test recursive directory search."""
        # Create nested structure
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        
        (tmp_path / "file1.txt").write_text("target\n")
        (subdir / "file2.txt").write_text("target\n")
        
        grep = GrepTool()
        results = grep.search(
            query="target",
            paths=[str(tmp_path)],
            mode=SearchMode.EXACT
        )
        
        assert len(results) == 2
    
    def test_gitignore_respected(self, tmp_path):
        """Test that .gitignore is respected."""
        # Create gitignore
        gitignore = tmp_path / ".gitignore"
        gitignore.write_text("*.log\n")
        
        # Create files
        (tmp_path / "test.txt").write_text("target\n")
        (tmp_path / "debug.log").write_text("target\n")
        
        config = SearchConfig(respect_gitignore=True)
        grep = GrepTool(config=config)
        
        results = grep.search(
            query="target",
            paths=[str(tmp_path)],
            mode=SearchMode.EXACT
        )
        
        # Should only find test.txt, not debug.log
        assert len(results) == 1
        assert "test.txt" in results[0].file_path


class TestResultFormatting:
    """Test result formatting and serialization."""
    
    def test_result_to_dict(self, tmp_path):
        """Test SearchResult serialization."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\n")
        
        grep = GrepTool()
        results = grep.search(
            query="hello",
            paths=[str(tmp_path)],
            mode=SearchMode.EXACT
        )
        
        assert len(results) > 0
        result_dict = results[0].to_dict()
        
        assert "file_path" in result_dict
        assert "line_number" in result_dict
        assert "line_content" in result_dict
        assert "score" in result_dict
    
    def test_result_json(self, tmp_path):
        """Test SearchResult JSON output."""
        import json
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\n")
        
        grep = GrepTool()
        results = grep.search(
            query="hello",
            paths=[str(tmp_path)],
            mode=SearchMode.EXACT
        )
        
        json_str = results[0].to_json()
        parsed = json.loads(json_str)
        
        assert parsed["match_text"] == "hello"


class TestLangChainIntegration:
    """Test LangChain tool integration."""
    
    def test_tool_creation(self):
        """Test that LangChain tool can be created."""
        tool = LLMGrepTool(
            root_path="/workspace",
            default_mode="exact"
        )
        
        assert tool.name == "llm_grep"
        assert tool.root_path == "/workspace"
    
    def test_tool_execution(self, tmp_path):
        """Test LangChain tool execution."""
        # Create test file
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\nfoo bar\n")
        
        tool = LLMGrepTool(root_path=str(tmp_path))
        result = tool._run(
            query="hello",
            paths=".",
            mode="exact",
            max_results=5
        )
        
        assert "Found" in result
        assert "hello" in result
    
    def test_tool_no_results(self, tmp_path):
        """Test LangChain tool with no matches."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\n")
        
        tool = LLMGrepTool(root_path=str(tmp_path))
        result = tool._run(
            query="notfound",
            paths=".",
            mode="exact"
        )
        
        assert "No results" in result


class TestSearchModes:
    """Test different search modes."""
    
    def test_all_modes_available(self, tmp_path):
        """Test that all search modes work."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world\n")
        
        grep = GrepTool()
        
        for mode in [SearchMode.EXACT, SearchMode.REGEX]:
            results = grep.search(
                query="hello",
                paths=[str(tmp_path)],
                mode=mode
            )
            assert isinstance(results, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
