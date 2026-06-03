"""
Example usage of LLM GrepTool with LangChain.

This file demonstrates various ways to use the LLM GrepTool
with LangChain agents and chains.
"""

import os
from pathlib import Path


def example_basic_usage():
    """Basic usage of the GrepTool without LangChain."""
    print("=" * 80)
    print("EXAMPLE 1: Basic GrepTool Usage")
    print("=" * 80)
    
    from llm_greptool import GrepTool, SearchMode
    
    # Create a grep tool instance
    grep = GrepTool()
    
    # Search for Python files in current directory
    results = grep.search(
        query="def ",
        paths=["."],
        mode=SearchMode.REGEX,
        max_results=5
    )
    
    print(f"\nFound {len(results)} matches for 'def ' (regex)")
    for result in results[:3]:
        print(f"  - {result.file_path}:{result.line_number}")
        print(f"    {result.line_content.strip()}")
    
    return results


def example_langchain_tool():
    """Using LLMGrepTool as a LangChain tool."""
    print("\n" + "=" * 80)
    print("EXAMPLE 2: LangChain Tool Integration")
    print("=" * 80)
    
    from llm_greptool import LangChainGrepTool
    
    # Create a LangChain-compatible grep tool
    grep_tool = LangChainGrepTool(
        root_path="/workspace",
        default_mode="hybrid"
    )
    
    # Use the tool directly
    result = grep_tool._run(
        query="import",
        paths=".",
        mode="regex",
        max_results=3
    )
    
    print(result)
    return grep_tool


def example_semantic_search():
    """Example of semantic search using embeddings."""
    print("\n" + "=" * 80)
    print("EXAMPLE 3: Semantic Search")
    print("=" * 80)
    
    from llm_greptool import GrepTool, SearchMode
    
    grep = GrepTool()
    
    # Semantic search finds code by meaning, not just text
    results = grep.search(
        query="function that reads files",
        paths=["."],
        mode=SearchMode.SEMANTIC,
        max_results=3
    )
    
    print(f"\nSemantic search for 'function that reads files':")
    for result in results:
        print(f"  Score: {result.score:.3f} | {result.file_path}:{result.line_number}")
        print(f"    {result.line_content.strip()}")
    
    return results


def example_create_multiple_tools():
    """Create multiple grep tools for a LangChain agent."""
    print("\n" + "=" * 80)
    print("EXAMPLE 4: Multiple Tools for Agent")
    print("=" * 80)
    
    from llm_greptool.langchain_integration import create_grep_tools
    
    # Create a suite of grep tools
    tools = create_grep_tools(
        root_path="/workspace",
        enable_semantic=True,
        enable_regex=True,
        embedding_model="all-MiniLM-L6-v2"
    )
    
    print(f"\nCreated {len(tools)} tools:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:80]}...")
    
    return tools


def example_agent_integration():
    """Full example with LangChain agent (requires OpenAI API key)."""
    print("\n" + "=" * 80)
    print("EXAMPLE 5: LangChain Agent Integration")
    print("=" * 80)
    
    try:
        from langchain.agents import initialize_agent, AgentType
        from langchain_openai import ChatOpenAI
        from llm_greptool.langchain_integration import create_grep_tools
        
        # Check for API key
        if not os.getenv("OPENAI_API_KEY"):
            print("\n[SKIP] OPENAI_API_KEY not set. Skipping agent example.")
            print("Set your API key and run again to see full agent integration.")
            return None
        
        # Create tools
        tools = create_grep_tools(root_path="/workspace")
        
        # Initialize LLM
        llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        
        # Create agent
        agent = initialize_agent(
            tools=tools,
            llm=llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )
        
        # Run a query
        print("\nAsking agent to find Python function definitions...")
        response = agent.run("Find all Python function definitions in the codebase")
        
        print("\nAgent response:")
        print(response)
        
        return agent
        
    except ImportError as e:
        print(f"\n[SKIP] Missing dependency: {e}")
        print("Install with: pip install langchain-openai")
        return None
    except Exception as e:
        print(f"\n[ERROR] Agent integration failed: {e}")
        return None


def example_json_output():
    """Example of getting structured JSON output."""
    print("\n" + "=" * 80)
    print("EXAMPLE 6: JSON Output")
    print("=" * 80)
    
    from llm_greptool import GrepTool, SearchMode
    import json
    
    grep = GrepTool()
    
    results = grep.search(
        query="class.*:",
        paths=["."],
        mode=SearchMode.REGEX,
        max_results=3
    )
    
    # Convert to JSON
    json_output = {
        "query": "class.*:",
        "count": len(results),
        "results": [r.to_dict() for r in results]
    }
    
    print("\nJSON output:")
    print(json.dumps(json_output, indent=2)[:500] + "...")
    
    return json_output


def example_context_windows():
    """Example showing context windows around matches."""
    print("\n" + "=" * 80)
    print("EXAMPLE 7: Context Windows")
    print("=" * 80)
    
    from llm_greptool import GrepTool, SearchConfig, SearchMode
    
    # Configure with larger context
    config = SearchConfig(
        mode=SearchMode.EXACT,
        context_lines=3,  # 3 lines before and after
        max_results=2
    )
    
    grep = GrepTool(config=config)
    
    results = grep.search(
        query="LLMGrepTool",
        paths=["."],
    )
    
    print(f"\nSearching for 'LLMGrepTool' with 3 lines of context:\n")
    for result in results:
        print(f"File: {result.file_path}:{result.line_number}")
        print("-" * 60)
        
        # Print context before
        for line in result.context_before:
            print(f"  {line}")
        
        # Print match
        print(f"→ {result.line_content}")
        
        # Print context after
        for line in result.context_after:
            print(f"  {line}")
        
        print("-" * 60)
        print()
    
    return results


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("LLM GREPTOOL - USAGE EXAMPLES")
    print("=" * 80 + "\n")
    
    # Run examples
    example_basic_usage()
    example_langchain_tool()
    example_semantic_search()
    example_create_multiple_tools()
    example_agent_integration()
    example_json_output()
    example_context_windows()
    
    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
