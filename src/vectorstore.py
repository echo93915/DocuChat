"""Vector store abstraction for FAISS and Chroma backends."""

import logging
import json
import pickle
from pathlib import Path
from typing import List, Optional, Dict, Any
from datetime import datetime

import numpy as np

# Import vector store backends
try:
    import faiss
    FAISS_AVAILABLE = True
except ImportError:
    FAISS_AVAILABLE = False
    logging.warning("FAISS not available")

try:
    import chromadb
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    logging.warning("ChromaDB not available")

from .settings import settings
from .types import DocumentChunk, IndexStats
from .llm import embed_texts  # Using real OpenAI API

logger = logging.getLogger(__name__)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""
    pass


class BaseVectorStore:
    """Abstract base class for vector store implementations."""
    
    def __init__(self, index_dir: str):
        self.index_dir = Path(index_dir)
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.chunks: List[str] = []
        self.embeddings: Optional[np.ndarray] = None
        
    def build_index(self, chunks: List[str]) -> None:
        """Build vector index from text chunks."""
        raise NotImplementedError
    
    def search(self, query: str, k: int = 4) -> List[str]:
        """Search for similar chunks."""
        raise NotImplementedError
    
    def save_index(self) -> None:
        """Save index to disk."""
        raise NotImplementedError
    
    def load_index(self) -> bool:
        """Load index from disk. Returns True if successful."""
        raise NotImplementedError
    
    def get_stats(self) -> IndexStats:
        """Get index statistics."""
        raise NotImplementedError


class FAISSVectorStore(BaseVectorStore):
    """FAISS-based vector store implementation."""
    
    def __init__(self, index_dir: str):
        if not FAISS_AVAILABLE:
            raise VectorStoreError("FAISS is not available. Install with: pip install faiss-cpu")
        
        super().__init__(index_dir)
        self.index: Optional[faiss.Index] = None
        self.dimension = None  # Will be determined dynamically from first embedding
        
    def build_index(self, chunks: List[str]) -> None:
        """
        Build FAISS index from text chunks.
        
        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            raise VectorStoreError("No chunks provided for indexing")
        
        logger.info(f"Building FAISS index for {len(chunks)} chunks")
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embed_texts(chunks)
        
        # Filter out empty embeddings
        valid_embeddings = []
        valid_chunks = []
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:  # Skip empty embeddings
                valid_embeddings.append(embedding)
                valid_chunks.append(chunk)
            else:
                logger.warning(f"Skipping chunk {i} with empty embedding")
        
        if not valid_embeddings:
            raise VectorStoreError("No valid embeddings generated")
        
        # Convert to numpy array
        self.embeddings = np.array(valid_embeddings, dtype=np.float32)
        self.chunks = valid_chunks
        
        # Determine dimension from first embedding
        new_dimension = self.embeddings.shape[1]
        
        # Check if we're changing dimensions (warn user)
        if self.dimension is not None and self.dimension != new_dimension:
            logger.warning(f"Dimension change detected: {self.dimension}d → {new_dimension}d. "
                         f"Rebuilding index with new embedding provider.")
        
        self.dimension = new_dimension
        logger.info(f"Creating FAISS index with {len(valid_embeddings)} embeddings, dimension {self.dimension}")
        
        # Create FAISS index (inner product for cosine similarity)
        self.index = faiss.IndexFlatIP(self.dimension)
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        
        # Add embeddings to index
        self.index.add(self.embeddings)
        
        logger.info(f"FAISS index built successfully with {self.index.ntotal} vectors")
        
        # Save to disk
        self.save_index()
    
    def search(self, query: str, k: int = 4) -> List[str]:
        """
        Search for similar chunks using FAISS.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar chunk texts
        """
        if self.index is None:
            if not self.load_index():
                raise VectorStoreError("No index available. Build index first.")
        
        if k > len(self.chunks):
            k = len(self.chunks)
            logger.warning(f"Requested k={k} but only {len(self.chunks)} chunks available")
        
        logger.info(f"Searching FAISS index for query: {query[:50]}...")
        
        # Generate query embedding
        query_embeddings = embed_texts([query])
        if not query_embeddings or not query_embeddings[0]:
            raise VectorStoreError("Failed to generate query embedding")
        
        # Check dimension compatibility
        query_dim = len(query_embeddings[0])
        if query_dim != self.dimension:
            logger.error(f"Dimension mismatch: query embedding ({query_dim}d) vs index ({self.dimension}d)")
            raise VectorStoreError(
                f"Embedding dimension mismatch. Current provider generates {query_dim}d embeddings, "
                f"but existing index expects {self.dimension}d. Please rebuild the index or switch "
                f"to the original embedding provider."
            )
        
        query_vector = np.array([query_embeddings[0]], dtype=np.float32)
        faiss.normalize_L2(query_vector)
        
        # Search index
        similarities, indices = self.index.search(query_vector, k)
        
        # Return corresponding chunks
        results = []
        for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
            if idx < len(self.chunks):
                results.append(self.chunks[idx])
                logger.debug(f"Result {i+1}: similarity={similarity:.3f}, chunk_length={len(self.chunks[idx])}")
        
        logger.info(f"Found {len(results)} similar chunks")
        return results
    
    def save_index(self) -> None:
        """Save FAISS index and metadata to disk."""
        if self.index is None:
            raise VectorStoreError("No index to save")
        
        logger.info(f"Saving FAISS index to {self.index_dir}")
        
        # Save FAISS index
        index_path = self.index_dir / "faiss.index"
        faiss.write_index(self.index, str(index_path))
        
        # Save chunks
        chunks_path = self.index_dir / "chunks.pkl"
        with open(chunks_path, 'wb') as f:
            pickle.dump(self.chunks, f)
        
        # Save embeddings
        embeddings_path = self.index_dir / "embeddings.npy"
        np.save(embeddings_path, self.embeddings)
        
        # Save metadata
        metadata = {
            'total_chunks': len(self.chunks),
            'dimension': self.dimension,
            'created_at': datetime.now().isoformat(),
            'vector_store_type': 'faiss'
        }
        metadata_path = self.index_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("FAISS index saved successfully")
    
    def load_index(self) -> bool:
        """
        Load FAISS index from disk.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            index_path = self.index_dir / "faiss.index"
            chunks_path = self.index_dir / "chunks.pkl"
            embeddings_path = self.index_dir / "embeddings.npy"
            metadata_path = self.index_dir / "metadata.json"
            
            if not all(p.exists() for p in [index_path, chunks_path, embeddings_path]):
                logger.warning("FAISS index files not found")
                return False
            
            logger.info(f"Loading FAISS index from {self.index_dir}")
            
            # Load metadata to get dimension
            if metadata_path.exists():
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    self.dimension = metadata.get('dimension', 768)  # Default to Gemini dimension
            else:
                # Fallback: infer dimension from embeddings
                embeddings = np.load(embeddings_path)
                self.dimension = embeddings.shape[1]
            
            # Load FAISS index
            self.index = faiss.read_index(str(index_path))
            
            # Load chunks
            with open(chunks_path, 'rb') as f:
                self.chunks = pickle.load(f)
            
            # Load embeddings
            self.embeddings = np.load(embeddings_path)
            
            logger.info(f"FAISS index loaded: {len(self.chunks)} chunks, {self.index.ntotal} vectors, dimension {self.dimension}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load FAISS index: {e}")
            return False
    
    def get_stats(self) -> IndexStats:
        """Get FAISS index statistics."""
        if self.index is None and not self.load_index():
            return IndexStats(
                total_chunks=0,
                index_size_mb=0.0,
                embedding_dimension=768,  # Default dimension for Gemini embeddings
                vector_store_type="faiss",
                last_updated=datetime.now()
            )
        
        # Calculate index size
        index_size = 0
        for file_name in ["faiss.index", "chunks.pkl", "embeddings.npy", "metadata.json"]:
            file_path = self.index_dir / file_name
            if file_path.exists():
                index_size += file_path.stat().st_size
        
        return IndexStats(
            total_chunks=len(self.chunks),
            index_size_mb=index_size / (1024 * 1024),
            embedding_dimension=self.dimension or 768,  # Default to Gemini dimension
            vector_store_type="faiss",
            last_updated=datetime.now()
        )


class ChromaVectorStore(BaseVectorStore):
    """ChromaDB-based vector store implementation."""
    
    def __init__(self, index_dir: str):
        if not CHROMA_AVAILABLE:
            raise VectorStoreError("ChromaDB is not available. Install with: pip install chromadb")
        
        super().__init__(index_dir)
        self.client = chromadb.PersistentClient(path=str(self.index_dir))
        self.collection_name = "docuchat_collection"
        self.collection = None
        
    def build_index(self, chunks: List[str]) -> None:
        """
        Build ChromaDB collection from text chunks.
        
        Args:
            chunks: List of text chunks to index
        """
        if not chunks:
            raise VectorStoreError("No chunks provided for indexing")
        
        logger.info(f"Building ChromaDB collection for {len(chunks)} chunks")
        
        # Create or get collection
        try:
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"description": "DocuChat document chunks"}
            )
        except Exception:
            # Collection might already exist
            self.collection = self.client.get_collection(name=self.collection_name)
            # Clear existing data
            self.collection.delete()
        
        # Generate embeddings
        logger.info("Generating embeddings...")
        embeddings = embed_texts(chunks)
        
        # Filter out empty embeddings and prepare data
        valid_chunks = []
        valid_embeddings = []
        valid_ids = []
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            if embedding:  # Skip empty embeddings
                valid_chunks.append(chunk)
                valid_embeddings.append(embedding)
                valid_ids.append(f"chunk_{i}")
            else:
                logger.warning(f"Skipping chunk {i} with empty embedding")
        
        if not valid_embeddings:
            raise VectorStoreError("No valid embeddings generated")
        
        # Add to collection
        self.collection.add(
            embeddings=valid_embeddings,
            documents=valid_chunks,
            ids=valid_ids
        )
        
        self.chunks = valid_chunks
        logger.info(f"ChromaDB collection built successfully with {len(valid_chunks)} documents")
    
    def search(self, query: str, k: int = 4) -> List[str]:
        """
        Search for similar chunks using ChromaDB.
        
        Args:
            query: Search query
            k: Number of results to return
            
        Returns:
            List of similar chunk texts
        """
        if self.collection is None:
            if not self.load_index():
                raise VectorStoreError("No collection available. Build index first.")
        
        logger.info(f"Searching ChromaDB collection for query: {query[:50]}...")
        
        # Generate query embedding
        query_embeddings = embed_texts([query])
        if not query_embeddings or not query_embeddings[0]:
            raise VectorStoreError("Failed to generate query embedding")
        
        # Search collection
        results = self.collection.query(
            query_embeddings=[query_embeddings[0]],
            n_results=min(k, len(self.chunks))
        )
        
        documents = results['documents'][0] if results['documents'] else []
        logger.info(f"Found {len(documents)} similar chunks")
        
        return documents
    
    def save_index(self) -> None:
        """Save ChromaDB collection (automatically persisted)."""
        logger.info("ChromaDB collection automatically persisted")
    
    def load_index(self) -> bool:
        """
        Load ChromaDB collection.
        
        Returns:
            True if loaded successfully, False otherwise
        """
        try:
            self.collection = self.client.get_collection(name=self.collection_name)
            
            # Get all documents to populate chunks list
            all_docs = self.collection.get()
            self.chunks = all_docs['documents'] if all_docs['documents'] else []
            
            logger.info(f"ChromaDB collection loaded: {len(self.chunks)} chunks")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ChromaDB collection: {e}")
            return False
    
    def get_stats(self) -> IndexStats:
        """Get ChromaDB collection statistics."""
        if self.collection is None and not self.load_index():
            return IndexStats(
                total_chunks=0,
                index_size_mb=0.0,
                embedding_dimension=1536,
                vector_store_type="chroma",
                last_updated=datetime.now()
            )
        
        # Calculate directory size
        index_size = sum(
            f.stat().st_size for f in self.index_dir.rglob('*') if f.is_file()
        )
        
        return IndexStats(
            total_chunks=len(self.chunks),
            index_size_mb=index_size / (1024 * 1024),
            embedding_dimension=1536,
            vector_store_type="chroma",
            last_updated=datetime.now()
        )


def get_vector_store(vector_store_type: Optional[str] = None) -> BaseVectorStore:
    """
    Factory function to get the appropriate vector store instance.
    
    Args:
        vector_store_type: Type of vector store ('faiss' or 'chroma')
        
    Returns:
        Vector store instance
    """
    store_type = vector_store_type or settings.vector_store
    index_dir = settings.index_dir
    
    if store_type == "faiss":
        return FAISSVectorStore(index_dir)
    elif store_type == "chroma":
        return ChromaVectorStore(index_dir)
    else:
        raise VectorStoreError(f"Unsupported vector store type: {store_type}")


# Convenience functions for the main interface
def build_index(chunks: List[str]) -> None:
    """Build vector index from chunks using configured vector store."""
    store = get_vector_store()
    store.build_index(chunks)


def search(query: str, k: int = 4) -> List[str]:
    """Search for similar chunks using configured vector store."""
    store = get_vector_store()
    return store.search(query, k)
