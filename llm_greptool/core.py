"""
Core search functionality for LLM GrepTool.

Provides exact matching, regex search, and semantic search capabilities.
"""

import os
import re
import gzip
import zipfile
import tarfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Iterator
from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib


class SearchMode(Enum):
    """Search mode enumeration."""
    EXACT = "exact"           # Fixed string matching
    REGEX = "regex"           # Regular expression matching
    FUZZY = "fuzzy"           # Approximate matching (edit distance)
    SEMANTIC = "semantic"     # Semantic/embedding-based search
    HYBRID = "hybrid"         # Combined exact + semantic


@dataclass
class SearchResult:
    """Represents a single search result."""
    file_path: str
    line_number: int
    line_content: str
    match_text: str
    context_before: List[str] = field(default_factory=list)
    context_after: List[str] = field(default_factory=list)
    score: float = 1.0  # Relevance score (1.0 for exact matches)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "file_path": self.file_path,
            "line_number": self.line_number,
            "line_content": self.line_content,
            "match_text": self.match_text,
            "context_before": self.context_before,
            "context_after": self.context_after,
            "score": self.score,
            "metadata": self.metadata,
        }
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)


@dataclass
class SearchConfig:
    """Configuration for search operations."""
    mode: SearchMode = SearchMode.EXACT
    recursive: bool = True
    case_sensitive: bool = False
    context_lines: int = 2
    max_results: int = 100
    include_patterns: List[str] = field(default_factory=list)
    exclude_patterns: List[str] = field(default_factory=list)
    respect_gitignore: bool = True
    search_archives: bool = False
    search_compressed: bool = True
    parallel: bool = True
    num_threads: int = 4
    
    # Semantic search config
    embedding_model: str = "all-MiniLM-L6-v2"
    top_k_semantic: int = 10
    
    # Fuzzy search config
    max_edit_distance: int = 2


class GrepTool:
    """
    Main GrepTool class providing powerful search capabilities.
    
    Supports:
    - Exact string matching
    - Regular expression search
    - Fuzzy/approximate matching
    - Semantic search with embeddings
    - Hybrid search modes
    - Multi-file and archive search
    - Context windows
    - Parallel processing
    """
    
    def __init__(self, config: Optional[SearchConfig] = None):
        """Initialize the GrepTool with optional configuration."""
        self.config = config or SearchConfig()
        self._index_cache: Dict[str, List[str]] = {}
        self._embedding_index: Optional[Any] = None
        self._embeddings = None
        
    def search(
        self,
        query: str,
        paths: Union[str, List[str]],
        mode: Optional[SearchMode] = None,
        **kwargs
    ) -> List[SearchResult]:
        """
        Perform a search operation.
        
        Args:
            query: Search query (string, regex pattern, or natural language)
            paths: File path(s) or directory path(s) to search
            mode: Search mode (overrides config if provided)
            **kwargs: Additional search parameters
            
        Returns:
            List of SearchResult objects
        """
        mode = mode or self.config.mode
        
        # Normalize paths
        if isinstance(paths, str):
            paths = [paths]
        
        # Collect all files to search
        files = self._collect_files(paths)
        
        # Execute search based on mode
        if mode == SearchMode.EXACT:
            results = self._exact_search(query, files, **kwargs)
        elif mode == SearchMode.REGEX:
            results = self._regex_search(query, files, **kwargs)
        elif mode == SearchMode.FUZZY:
            results = self._fuzzy_search(query, files, **kwargs)
        elif mode == SearchMode.SEMANTIC:
            results = self._semantic_search(query, files, **kwargs)
        elif mode == SearchMode.HYBRID:
            results = self._hybrid_search(query, files, **kwargs)
        else:
            raise ValueError(f"Unknown search mode: {mode}")
        
        # Sort by score (descending) and limit results
        results.sort(key=lambda r: (-r.score, r.file_path, r.line_number))
        max_results = kwargs.get('max_results', self.config.max_results)
        return results[:max_results]
    
    def _collect_files(self, paths: List[str]) -> List[Path]:
        """Collect all files to search from given paths."""
        files = []
        
        for path_str in paths:
            path = Path(path_str)
            
            if not path.exists():
                continue
                
            if path.is_file():
                files.append(path)
            elif path.is_dir():
                files.extend(self._walk_directory(path))
        
        return files
    
    def _walk_directory(self, root: Path) -> List[Path]:
        """Walk directory tree, respecting gitignore and filters."""
        files = []
        gitignore_patterns = []
        
        # Load .gitignore if present and configured
        if self.config.respect_gitignore:
            gitignore_file = root / ".gitignore"
            if gitignore_file.exists():
                gitignore_patterns = self._parse_gitignore(gitignore_file)
        
        for dirpath, dirnames, filenames in os.walk(root):
            current_path = Path(dirpath)
            
            # Filter directories
            dirnames[:] = [
                d for d in dirnames
                if not self._should_ignore(current_path / d, gitignore_patterns)
            ]
            
            # Filter files
            for filename in filenames:
                file_path = current_path / filename
                
                if self._should_ignore(file_path, gitignore_patterns):
                    continue
                
                if self._matches_include_exclude(file_path):
                    files.append(file_path)
        
        return files
    
    def _parse_gitignore(self, gitignore_path: Path) -> List[str]:
        """Parse .gitignore file and return patterns."""
        patterns = []
        try:
            with open(gitignore_path, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith('#'):
                        patterns.append(line)
        except Exception:
            pass
        return patterns
    
    def _should_ignore(self, path: Path, gitignore_patterns: List[str]) -> bool:
        """Check if path should be ignored based on gitignore patterns."""
        name = path.name
        rel_path = str(path)
        
        # Check against gitignore patterns
        for pattern in gitignore_patterns:
            if pattern.startswith('/'):
                # Absolute pattern
                if rel_path.endswith(pattern[1:]):
                    return True
            elif pattern.endswith('/'):
                # Directory pattern
                if path.is_dir() and name == pattern[:-1]:
                    return True
            elif '*' in pattern:
                # Glob pattern
                import fnmatch
                if fnmatch.fnmatch(name, pattern):
                    return True
            else:
                # Simple pattern
                if name == pattern or rel_path.endswith('/' + pattern):
                    return True
        
        return False
    
    def _matches_include_exclude(self, path: Path) -> bool:
        """Check if path matches include/exclude patterns."""
        import fnmatch
        name = path.name
        
        # Check exclude patterns
        for pattern in self.config.exclude_patterns:
            if fnmatch.fnmatch(name, pattern):
                return False
        
        # Check include patterns (if specified)
        if self.config.include_patterns:
            for pattern in self.config.include_patterns:
                if fnmatch.fnmatch(name, pattern):
                    return True
            return False
        
        return True
    
    def _read_file_content(self, file_path: Path) -> Optional[str]:
        """Read file content, handling compressed files."""
        try:
            # Check file extension
            suffix = file_path.suffix.lower()
            
            if suffix == '.gz':
                with gzip.open(file_path, 'rt', encoding='utf-8', errors='ignore') as f:
                    return f.read()
            elif suffix in ['.zip', '.tar', '.bz2', '.xz']:
                # Handle archives if configured
                if self.config.search_archives:
                    return self._extract_archive_content(file_path)
                return None
            else:
                # Regular file
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    return f.read()
        except Exception:
            return None
    
    def _extract_archive_content(self, archive_path: Path) -> Optional[str]:
        """Extract text content from archive files."""
        try:
            suffix = archive_path.suffix.lower()
            
            if suffix == '.zip':
                with zipfile.ZipFile(archive_path, 'r') as zf:
                    contents = []
                    for name in zf.namelist():
                        try:
                            with zf.open(name) as f:
                                contents.append(f.read().decode('utf-8', errors='ignore'))
                        except Exception:
                            continue
                    return '\n'.join(contents)
            
            elif suffix == '.tar' or suffix.endswith('.tar.gz') or suffix.endswith('.tgz'):
                with tarfile.open(archive_path, 'r:*') as tf:
                    contents = []
                    for member in tf.getmembers():
                        if member.isfile():
                            try:
                                f = tf.extractfile(member)
                                if f:
                                    contents.append(f.read().decode('utf-8', errors='ignore'))
                            except Exception:
                                continue
                    return '\n'.join(contents)
            
        except Exception:
            pass
        
        return None
    
    def _exact_search(
        self,
        query: str,
        files: List[Path],
        **kwargs
    ) -> List[SearchResult]:
        """Perform exact string matching."""
        results = []
        case_sensitive = kwargs.get('case_sensitive', self.config.case_sensitive)
        
        if not case_sensitive:
            query_lower = query.lower()
        
        for file_path in files:
            content = self._read_file_content(file_path)
            if content is None:
                continue
            
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                search_line = line if case_sensitive else line.lower()
                
                if query_lower in search_line if not case_sensitive else query in line:
                    context_before, context_after = self._get_context(lines, line_num - 1)
                    
                    results.append(SearchResult(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line,
                        match_text=query,
                        context_before=context_before,
                        context_after=context_after,
                        score=1.0,
                        metadata={'match_type': 'exact'}
                    ))
        
        return results
    
    def _regex_search(
        self,
        pattern: str,
        files: List[Path],
        **kwargs
    ) -> List[SearchResult]:
        """Perform regular expression search."""
        results = []
        
        try:
            flags = 0 if kwargs.get('case_sensitive', self.config.case_sensitive) else re.IGNORECASE
            regex = re.compile(pattern, flags)
        except re.error as e:
            raise ValueError(f"Invalid regex pattern: {e}")
        
        for file_path in files:
            content = self._read_file_content(file_path)
            if content is None:
                continue
            
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                match = regex.search(line)
                if match:
                    context_before, context_after = self._get_context(lines, line_num - 1)
                    
                    results.append(SearchResult(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line,
                        match_text=match.group(0),
                        context_before=context_before,
                        context_after=context_after,
                        score=1.0,
                        metadata={'match_type': 'regex', 'pattern': pattern}
                    ))
        
        return results
    
    def _fuzzy_search(
        self,
        query: str,
        files: List[Path],
        **kwargs
    ) -> List[SearchResult]:
        """Perform fuzzy/approximate string matching."""
        results = []
        max_distance = kwargs.get('max_edit_distance', self.config.max_edit_distance)
        
        for file_path in files:
            content = self._read_file_content(file_path)
            if content is None:
                continue
            
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                # Simple fuzzy matching using substring with edit distance
                # For production, use a library like rapidfuzz
                distance = self._levenshtein_distance(query, line)
                
                if distance <= max_distance * len(query):
                    context_before, context_after = self._get_context(lines, line_num - 1)
                    score = 1.0 - (distance / max(len(query), len(line)))
                    
                    results.append(SearchResult(
                        file_path=str(file_path),
                        line_number=line_num,
                        line_content=line,
                        match_text=query,
                        context_before=context_before,
                        context_after=context_after,
                        score=max(0.0, score),
                        metadata={'match_type': 'fuzzy', 'edit_distance': distance}
                    ))
        
        return results
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein edit distance between two strings."""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def _semantic_search(
        self,
        query: str,
        files: List[Path],
        **kwargs
    ) -> List[SearchResult]:
        """Perform semantic search using embeddings."""
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("sentence-transformers required for semantic search. Install with: pip install sentence-transformers")
        
        # Initialize embedding model
        if self._embeddings is None:
            model_name = kwargs.get('embedding_model', self.config.embedding_model)
            self._embeddings = SentenceTransformer(model_name)
        
        top_k = kwargs.get('top_k', self.config.top_k_semantic)
        
        # Encode query
        query_embedding = self._embeddings.encode([query])[0]
        
        results = []
        
        # Process files in chunks for efficiency
        all_chunks = []
        chunk_metadata = []
        
        for file_path in files:
            content = self._read_file_content(file_path)
            if content is None:
                continue
            
            # Split into chunks (lines or paragraphs)
            lines = content.split('\n')
            
            for line_num, line in enumerate(lines, 1):
                if len(line.strip()) > 10:  # Skip very short lines
                    all_chunks.append(line)
                    chunk_metadata.append({
                        'file_path': str(file_path),
                        'line_number': line_num,
                        'full_content': content
                    })
        
        if not all_chunks:
            return results
        
        # Encode all chunks
        chunk_embeddings = self._embeddings.encode(all_chunks)
        
        # Calculate similarity scores
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
        
        # Get top-k results
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        for idx in top_indices:
            if similarities[idx] < 0.3:  # Threshold for relevance
                continue
            
            meta = chunk_metadata[idx]
            lines = meta['full_content'].split('\n')
            line_num = meta['line_number']
            context_before, context_after = self._get_context(lines, line_num - 1)
            
            results.append(SearchResult(
                file_path=meta['file_path'],
                line_number=line_num,
                line_content=all_chunks[idx],
                match_text=all_chunks[idx],
                context_before=context_before,
                context_after=context_after,
                score=float(similarities[idx]),
                metadata={'match_type': 'semantic', 'embedding_model': self._embeddings.model_card_data.model_name}
            ))
        
        return results
    
    def _hybrid_search(
        self,
        query: str,
        files: List[Path],
        **kwargs
    ) -> List[SearchResult]:
        """Perform hybrid search combining exact and semantic matching."""
        # First do exact/regex search
        exact_results = self._exact_search(query, files, **kwargs)
        
        # Then do semantic search
        try:
            semantic_results = self._semantic_search(query, files, **kwargs)
        except Exception:
            semantic_results = []
        
        # Combine and deduplicate
        seen = set()
        combined = []
        
        for result in exact_results + semantic_results:
            key = (result.file_path, result.line_number)
            if key not in seen:
                seen.add(key)
                # Boost exact matches slightly
                if result.metadata.get('match_type') == 'exact':
                    result.score = min(1.0, result.score + 0.1)
                combined.append(result)
        
        return combined
    
    def _get_context(
        self,
        lines: List[str],
        line_idx: int,
        context_lines: Optional[int] = None
    ) -> tuple:
        """Get context lines before and after the match."""
        if context_lines is None:
            context_lines = self.config.context_lines
        
        start = max(0, line_idx - context_lines)
        end = min(len(lines), line_idx + context_lines + 1)
        
        context_before = lines[start:line_idx]
        context_after = lines[line_idx + 1:end]
        
        return context_before, context_after
    
    def search_generator(
        self,
        query: str,
        paths: Union[str, List[str]],
        mode: Optional[SearchMode] = None,
        **kwargs
    ) -> Iterator[SearchResult]:
        """Generator version of search for streaming results."""
        mode = mode or self.config.mode
        
        if isinstance(paths, str):
            paths = [paths]
        
        files = self._collect_files(paths)
        
        # Yield results as they are found
        if mode == SearchMode.EXACT:
            for result in self._exact_search(query, files, **kwargs):
                yield result
        elif mode == SearchMode.REGEX:
            for result in self._regex_search(query, files, **kwargs):
                yield result
        # Add other modes as needed

