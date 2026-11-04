"""Tests for LLM GrepTool."""

import pytest
import tempfile
import os
from pathlib import Path

from llm_greptool import GrepTool, SearchMode, SearchResult, SearchConfig
from llm_greptool.langchain_integration import LLMGrepTool, create_grep_tools


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


class TestRootPathConfinement:
    """Test that the LangChain tools only search inside root_path."""

    @pytest.fixture
    def root(self, tmp_path):
        """Create a root with one file, next to a directory that holds a secret."""
        root = tmp_path / "root"
        (root / "src").mkdir(parents=True)
        (root / "src" / "app.py").write_text("needle inside\n")
        (tmp_path / "outside").mkdir()
        (tmp_path / "outside" / "secret.txt").write_text("needle TOP_SECRET\n")
        return root

    def test_absolute_path_outside_root_is_rejected(self, root):
        """Test that an absolute path outside the root returns an error."""
        tool = LLMGrepTool(root_path=str(root))
        result = tool.invoke({"query": "needle", "paths": str(root.parent / "outside")})

        assert result.startswith("Error: path")
        assert "outside the search root" in result
        assert "TOP_SECRET" not in result

    def test_parent_directory_path_is_rejected(self, root):
        """Test that a path climbing out of the root with ../ returns an error."""
        tool = LLMGrepTool(root_path=str(root))
        result = tool.invoke({"query": "needle", "paths": "../outside"})

        assert result.startswith("Error: path")
        assert "TOP_SECRET" not in result

    def test_symlink_to_outside_is_rejected(self, root):
        """Test that a symlink inside the root that points outside returns an error."""
        try:
            (root / "link").symlink_to(root.parent / "outside", target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not supported on this system")

        tool = LLMGrepTool(root_path=str(root))
        result = tool.invoke({"query": "needle", "paths": "link"})

        assert result.startswith("Error: path")
        assert "TOP_SECRET" not in result

    def test_symlinked_file_inside_searched_directory_is_skipped(self, root):
        """Test that a symlinked file met while walking a directory is not read."""
        try:
            (root / "src" / "leak.txt").symlink_to(root.parent / "outside" / "secret.txt")
        except (OSError, NotImplementedError):
            pytest.skip("symlinks are not supported on this system")

        tool = LLMGrepTool(root_path=str(root))
        result = tool.invoke({"query": "needle"})

        assert result.startswith("Found 1 result(s)")
        assert "app.py" in result
        assert "TOP_SECRET" not in result

        # The core search skips the link too, like `grep -r`
        results = GrepTool().search("needle", paths=[str(root)], mode=SearchMode.EXACT)
        assert [r.line_content.strip() for r in results] == ["needle inside"]

    def test_relative_path_inside_root_is_searched(self, root):
        """Test that a relative path inside the root still works."""
        tool = LLMGrepTool(root_path=str(root))
        result = tool.invoke({"query": "needle", "paths": "src"})

        assert result.startswith("Found 1 result(s)")
        assert "app.py" in result

    def test_no_paths_searches_whole_root(self, root):
        """Test that leaving out paths searches root_path."""
        tool = LLMGrepTool(root_path=str(root))
        result = tool.invoke({"query": "needle"})

        assert result.startswith("Found 1 result(s)")
        assert "app.py" in result
        assert "TOP_SECRET" not in result

    def test_create_grep_tools_are_confined(self, root):
        """Test that every tool from create_grep_tools() rejects outside paths."""
        tools = create_grep_tools(root_path=str(root))

        assert len(tools) == 3
        for tool in tools:
            result = tool.invoke({"query": "needle", "paths": "../outside"})
            assert result.startswith("Error: path"), tool.name
            assert "TOP_SECRET" not in result


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
