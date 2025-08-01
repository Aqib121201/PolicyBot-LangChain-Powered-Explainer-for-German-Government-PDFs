"""
Vector Store for Document Retrieval.

Handles FAISS-based vector storage and similarity search for German government documents.
"""

import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import faiss
from sentence_transformers import SentenceTransformer
from loguru import logger

from .config import get_config


@dataclass
class SearchResult:
    """Represents a search result with metadata."""
    
    chunk_id: str
    text: str
    score: float
    page_number: int
    start_char: int
    end_char: int
    metadata: Dict[str, Any] = None


class VectorStore:
    """FAISS-based vector store for document retrieval."""
    
    def __init__(self, config=None):
        """Initialize vector store with configuration."""
        self.config = config or get_config()
        self.logger = logger.bind(name="VectorStore")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(self.config.model.embedding_model)
        
        # FAISS index
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        
        # Index parameters
        self.dimension = self.config.vector_store.dimension
        self.similarity_metric = self.config.vector_store.similarity_metric
        
        self.logger.info(f"Initialized VectorStore with {self.config.model.embedding_model}")
    
    def add_documents(self, chunks: List[Any], metadata: List[Dict[str, Any]] = None):
        """
        Add document chunks to the vector store.
        
        Args:
            chunks: List of text chunks (TextChunk objects or strings)
            metadata: Optional metadata for each chunk
        """
        if not chunks:
            self.logger.warning("No chunks provided for indexing")
            return
        
        self.logger.info(f"Adding {len(chunks)} chunks to vector store")
        
        # Extract text from chunks
        texts = []
        chunk_metadata = []
        
        for i, chunk in enumerate(chunks):
            if hasattr(chunk, 'text'):
                # TextChunk object
                texts.append(chunk.text)
                chunk_metadata.append({
                    'chunk_id': chunk.chunk_id,
                    'page_number': chunk.page_number,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'language': chunk.language,
                    'confidence': chunk.confidence
                })
            else:
                # String
                texts.append(str(chunk))
                chunk_metadata.append({
                    'chunk_id': f'chunk_{i:04d}',
                    'page_number': 1,
                    'start_char': 0,
                    'end_char': len(str(chunk)),
                    'language': self.config.processing.default_language,
                    'confidence': 1.0
                })
        
        # Generate embeddings
        embeddings = self.embedding_model.encode(
            texts,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Initialize or update FAISS index
        if self.index is None:
            self._initialize_index(embeddings.shape[1])
        
        # Add to index
        self.index.add(embeddings.astype('float32'))
        
        # Store chunks and metadata
        self.chunks.extend(texts)
        self.chunk_metadata.extend(chunk_metadata)
        
        self.logger.info(f"Successfully indexed {len(chunks)} chunks")
    
    def _initialize_index(self, dimension: int):
        """Initialize FAISS index."""
        if self.similarity_metric == "cosine":
            # Normalize vectors for cosine similarity
            self.index = faiss.IndexFlatIP(dimension)
        elif self.similarity_metric == "euclidean":
            self.index = faiss.IndexFlatL2(dimension)
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        self.logger.info(f"Initialized FAISS index with {self.similarity_metric} similarity")
    
    def search(self, query: str, n_results: int = None) -> List[SearchResult]:
        """
        Search for similar documents.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of SearchResult objects
        """
        if self.index is None or len(self.chunks) == 0:
            self.logger.warning("No documents indexed")
            return []
        
        n_results = n_results or self.config.vector_store.n_results
        
        # Encode query
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        
        # Search
        if self.similarity_metric == "cosine":
            # Normalize query for cosine similarity
            faiss.normalize_L2(query_embedding)
        
        scores, indices = self.index.search(
            query_embedding.astype('float32'),
            min(n_results, len(self.chunks))
        )
        
        # Convert to SearchResult objects
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                metadata = self.chunk_metadata[idx]
                result = SearchResult(
                    chunk_id=metadata['chunk_id'],
                    text=self.chunks[idx],
                    score=float(score),
                    page_number=metadata['page_number'],
                    start_char=metadata['start_char'],
                    end_char=metadata['end_char'],
                    metadata=metadata
                )
                results.append(result)
        
        # Filter by score threshold
        threshold = self.config.vector_store.score_threshold
        results = [r for r in results if r.score >= threshold]
        
        self.logger.info(f"Found {len(results)} results for query: {query[:50]}...")
        return results
    
    def similarity_search(self, query: str, n_results: int = None) -> List[Tuple[str, float]]:
        """
        Simple similarity search returning (text, score) tuples.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of (text, score) tuples
        """
        results = self.search(query, n_results)
        return [(result.text, result.score) for result in results]
    
    def get_relevant_chunks(self, query: str, n_results: int = None) -> List[str]:
        """
        Get relevant text chunks for a query.
        
        Args:
            query: Search query
            n_results: Number of results to return
            
        Returns:
            List of relevant text chunks
        """
        results = self.search(query, n_results)
        return [result.text for result in results]
    
    def batch_search(self, queries: List[str], n_results: int = None) -> List[List[SearchResult]]:
        """
        Perform batch search for multiple queries.
        
        Args:
            queries: List of search queries
            n_results: Number of results per query
            
        Returns:
            List of search result lists
        """
        if not queries:
            return []
        
        n_results = n_results or self.config.vector_store.n_results
        
        # Encode all queries
        query_embeddings = self.embedding_model.encode(
            queries,
            show_progress_bar=True,
            batch_size=32,
            convert_to_numpy=True
        )
        
        # Batch search
        if self.similarity_metric == "cosine":
            faiss.normalize_L2(query_embeddings)
        
        scores, indices = self.index.search(
            query_embeddings.astype('float32'),
            min(n_results, len(self.chunks))
        )
        
        # Convert to results
        all_results = []
        for query_idx, (query_scores, query_indices) in enumerate(zip(scores, indices)):
            query_results = []
            for score, idx in zip(query_scores, query_indices):
                if idx < len(self.chunks):
                    metadata = self.chunk_metadata[idx]
                    result = SearchResult(
                        chunk_id=metadata['chunk_id'],
                        text=self.chunks[idx],
                        score=float(score),
                        page_number=metadata['page_number'],
                        start_char=metadata['start_char'],
                        end_char=metadata['end_char'],
                        metadata=metadata
                    )
                    query_results.append(result)
            
            # Filter by threshold
            threshold = self.config.vector_store.score_threshold
            query_results = [r for r in query_results if r.score >= threshold]
            all_results.append(query_results)
        
        self.logger.info(f"Batch search completed for {len(queries)} queries")
        return all_results
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get vector store statistics."""
        stats = {
            'total_chunks': len(self.chunks),
            'index_size': self.index.ntotal if self.index else 0,
            'dimension': self.dimension,
            'similarity_metric': self.similarity_metric,
            'score_threshold': self.config.vector_store.score_threshold
        }
        
        if self.chunks:
            # Text statistics
            chunk_lengths = [len(chunk) for chunk in self.chunks]
            stats.update({
                'avg_chunk_length': np.mean(chunk_lengths),
                'min_chunk_length': np.min(chunk_lengths),
                'max_chunk_length': np.max(chunk_lengths),
                'total_characters': sum(chunk_lengths)
            })
        
        return stats
    
    def save(self, directory: str):
        """Save vector store to disk."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        
        # Save FAISS index
        if self.index:
            faiss.write_index(self.index, str(directory / "faiss_index.bin"))
        
        # Save chunks and metadata
        with open(directory / "chunks.pkl", 'wb') as f:
            pickle.dump(self.chunks, f)
        
        with open(directory / "metadata.pkl", 'wb') as f:
            pickle.dump(self.chunk_metadata, f)
        
        # Save configuration
        config_data = {
            'dimension': self.dimension,
            'similarity_metric': self.similarity_metric,
            'embedding_model': self.config.model.embedding_model
        }
        
        with open(directory / "config.pkl", 'wb') as f:
            pickle.dump(config_data, f)
        
        self.logger.info(f"Vector store saved to: {directory}")
    
    def load(self, directory: str):
        """Load vector store from disk."""
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Vector store directory not found: {directory}")
        
        # Load FAISS index
        index_path = directory / "faiss_index.bin"
        if index_path.exists():
            self.index = faiss.read_index(str(index_path))
        
        # Load chunks and metadata
        with open(directory / "chunks.pkl", 'rb') as f:
            self.chunks = pickle.load(f)
        
        with open(directory / "metadata.pkl", 'rb') as f:
            self.chunk_metadata = pickle.load(f)
        
        # Load configuration
        with open(directory / "config.pkl", 'rb') as f:
            config_data = pickle.load(f)
            self.dimension = config_data['dimension']
            self.similarity_metric = config_data['similarity_metric']
        
        self.logger.info(f"Vector store loaded from: {directory}")
        self.logger.info(f"Loaded {len(self.chunks)} chunks")
    
    def clear(self):
        """Clear all indexed documents."""
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.logger.info("Vector store cleared")
    
    def update_chunk(self, chunk_id: str, new_text: str):
        """Update a specific chunk in the vector store."""
        # Find the chunk
        for i, metadata in enumerate(self.chunk_metadata):
            if metadata['chunk_id'] == chunk_id:
                # Update text
                self.chunks[i] = new_text
                
                # Re-encode the chunk
                new_embedding = self.embedding_model.encode([new_text], convert_to_numpy=True)
                
                # Remove old embedding and add new one
                if self.index:
                    # Note: FAISS doesn't support direct updates, so we need to rebuild
                    # This is a simplified approach - in production, consider using a different index type
                    self.logger.warning("Chunk update requires index rebuild - consider using a different index type")
                
                break
        else:
            self.logger.warning(f"Chunk {chunk_id} not found")
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[SearchResult]:
        """Get a specific chunk by ID."""
        for i, metadata in enumerate(self.chunk_metadata):
            if metadata['chunk_id'] == chunk_id:
                return SearchResult(
                    chunk_id=metadata['chunk_id'],
                    text=self.chunks[i],
                    score=1.0,  # Perfect match
                    page_number=metadata['page_number'],
                    start_char=metadata['start_char'],
                    end_char=metadata['end_char'],
                    metadata=metadata
                )
        return None
    
    def export_to_csv(self, output_path: str):
        """Export vector store data to CSV for analysis."""
        import pandas as pd
        
        data = []
        for i, (chunk, metadata) in enumerate(zip(self.chunks, self.chunk_metadata)):
            data.append({
                'chunk_id': metadata['chunk_id'],
                'text': chunk,
                'page_number': metadata['page_number'],
                'start_char': metadata['start_char'],
                'end_char': metadata['end_char'],
                'language': metadata['language'],
                'confidence': metadata['confidence'],
                'length': len(chunk)
            })
        
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False, encoding='utf-8')
        self.logger.info(f"Vector store data exported to: {output_path}") 