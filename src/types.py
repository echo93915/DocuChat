"""Data types and structures for DocuChat."""

from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime


@dataclass
class DocumentChunk:
    """Represents a chunk of text from a document."""
    
    text: str
    chunk_id: int
    start_char: int
    end_char: int
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self):
        """Validate chunk data after initialization."""
        if not self.text.strip():
            raise ValueError("Chunk text cannot be empty")
        if self.start_char < 0 or self.end_char < 0:
            raise ValueError("Character positions must be non-negative")
        if self.start_char >= self.end_char:
            raise ValueError("start_char must be less than end_char")


@dataclass
class RetrievalResult:
    """Represents the result of a vector search retrieval."""
    
    chunk: DocumentChunk
    similarity_score: float
    rank: int
    
    def __post_init__(self):
        """Validate retrieval result data."""
        if not 0.0 <= self.similarity_score <= 1.0:
            raise ValueError("Similarity score must be between 0 and 1")
        if self.rank < 0:
            raise ValueError("Rank must be non-negative")


@dataclass
class QueryResponse:
    """Represents the complete response to a user query."""
    
    query: str
    answer: str
    sources: List[DocumentChunk]
    retrieval_results: List[RetrievalResult]
    timestamp: datetime
    model_used: str
    tokens_used: Optional[int] = None
    processing_time: Optional[float] = None
    
    def __post_init__(self):
        """Validate query response data."""
        if not self.query.strip():
            raise ValueError("Query cannot be empty")
        if not self.answer.strip():
            raise ValueError("Answer cannot be empty")
        if len(self.sources) != len(self.retrieval_results):
            raise ValueError("Sources and retrieval results must have same length")


@dataclass
class DocumentMetadata:
    """Metadata about an ingested document."""
    
    filename: str
    file_size: int
    num_pages: Optional[int]
    num_chunks: int
    ingestion_timestamp: datetime
    total_characters: int
    avg_chunk_size: float
    
    def __post_init__(self):
        """Validate document metadata."""
        if self.file_size <= 0:
            raise ValueError("File size must be positive")
        if self.num_chunks <= 0:
            raise ValueError("Number of chunks must be positive")
        if self.total_characters <= 0:
            raise ValueError("Total characters must be positive")


@dataclass
class IndexStats:
    """Statistics about the vector index."""
    
    total_chunks: int
    index_size_mb: float
    embedding_dimension: int
    vector_store_type: str
    last_updated: datetime
    
    def __post_init__(self):
        """Validate index statistics."""
        if self.total_chunks < 0:
            raise ValueError("Total chunks must be non-negative")
        if self.index_size_mb < 0:
            raise ValueError("Index size must be non-negative")
        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
