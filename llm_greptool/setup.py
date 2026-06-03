"""Setup script for LLM GrepTool."""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="llm-greptool",
    version="0.1.0",
    author="LLM GrepTool Team",
    description="AI-powered grep tool with LangChain integration",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "langchain>=0.1.0",
        "langchain-core>=0.1.0",
        "pydantic>=2.0.0",
        "rich>=13.0.0",
    ],
    extras_require={
        "semantic": [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "scikit-learn>=1.0.0",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=4.0.0",
            "black>=23.0.0",
            "ruff>=0.1.0",
        ],
        "full": [
            "sentence-transformers>=2.2.0",
            "faiss-cpu>=1.7.0",
            "scikit-learn>=1.0.0",
            "langchain-openai>=0.0.5",
        ],
    },
    entry_points={
        "console_scripts": [
            "llm-grep=llm_greptool.cli:main",
        ],
    },
)
