"""
Document chunking strategies for RAG QA System
"""

import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BaseChunker(ABC):
    """Abstract base class for document chunkers"""
    
    @abstractmethod
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text into smaller pieces"""
        pass


class FixedSizeChunker(BaseChunker):
    """Chunk text into fixed-size pieces"""
    
    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text into fixed-size pieces with overlap"""
        if not text.strip():
            return []
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # If this is not the last chunk, try to break at a word boundary
            if end < len(text):
                # Find the last space before the end
                last_space = text.rfind(' ', start, end)
                if last_space > start:
                    end = last_space
            
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunk_metadata = metadata.copy() if metadata else {}
                chunk_metadata.update({
                    'chunk_id': len(chunks),
                    'chunk_size': len(chunk_text),
                    'start_pos': start,
                    'end_pos': end
                })
                
                chunks.append({
                    'content': chunk_text,
                    'metadata': chunk_metadata
                })
            
            # Move start position for next chunk
            start = end - self.chunk_overlap
            if start >= len(text):
                break
        
        return chunks


class ParagraphChunker(BaseChunker):
    """Chunk text by paragraphs"""
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 2000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text by paragraphs"""
        if not text.strip():
            return []
        
        # Split by paragraph breaks
        paragraphs = re.split(r'\n\s*\n', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            # If adding this paragraph would exceed max size, save current chunk
            if current_chunk and len(current_chunk) + len(paragraph) > self.max_chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_size': len(current_chunk),
                        'chunk_type': 'paragraph'
                    })
                    
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': chunk_metadata
                    })
                    chunk_id += 1
                
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
        
        # Add the last chunk if it meets minimum size
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'chunk_size': len(current_chunk),
                'chunk_type': 'paragraph'
            })
            
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': chunk_metadata
            })
        
        return chunks


class SentenceChunker(BaseChunker):
    """Chunk text by sentences"""
    
    def __init__(self, min_chunk_size: int = 100, max_chunk_size: int = 1000):
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
    
    def chunk(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """Chunk text by sentences"""
        if not text.strip():
            return []
        
        # Simple sentence splitting (can be improved with NLP libraries)
        sentences = re.split(r'[.!?]+', text)
        chunks = []
        current_chunk = ""
        chunk_id = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # If adding this sentence would exceed max size, save current chunk
            if current_chunk and len(current_chunk) + len(sentence) > self.max_chunk_size:
                if len(current_chunk) >= self.min_chunk_size:
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata.update({
                        'chunk_id': chunk_id,
                        'chunk_size': len(current_chunk),
                        'chunk_type': 'sentence'
                    })
                    
                    chunks.append({
                        'content': current_chunk.strip(),
                        'metadata': chunk_metadata
                    })
                    chunk_id += 1
                
                current_chunk = sentence
            else:
                if current_chunk:
                    current_chunk += ". " + sentence
                else:
                    current_chunk = sentence
        
        # Add the last chunk if it meets minimum size
        if current_chunk and len(current_chunk) >= self.min_chunk_size:
            chunk_metadata = metadata.copy() if metadata else {}
            chunk_metadata.update({
                'chunk_id': chunk_id,
                'chunk_size': len(current_chunk),
                'chunk_type': 'sentence'
            })
            
            chunks.append({
                'content': current_chunk.strip(),
                'metadata': chunk_metadata
            })
        
        return chunks


class DocumentChunker:
    """Main document chunker that supports multiple strategies"""
    
    def __init__(self, strategy: str = "fixed_size", **kwargs):
        """
        Initialize chunker with specified strategy
        
        Args:
            strategy: Chunking strategy ("fixed_size", "paragraph", "sentence")
            **kwargs: Strategy-specific parameters
        """
        self.strategy = strategy
        
        if strategy == "fixed_size":
            self.chunker = FixedSizeChunker(**kwargs)
        elif strategy == "paragraph":
            self.chunker = ParagraphChunker(**kwargs)
        elif strategy == "sentence":
            self.chunker = SentenceChunker(**kwargs)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Chunk a single document"""
        content = document.get('content', '')
        metadata = document.get('metadata', {})
        
        return self.chunker.chunk(content, metadata)
    
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Chunk multiple documents"""
        all_chunks = []
        
        for doc in documents:
            chunks = self.chunk_document(doc)
            all_chunks.extend(chunks)
        
        return all_chunks
    
    def get_chunking_info(self) -> Dict[str, Any]:
        """Get information about the current chunking strategy"""
        return {
            'strategy': self.strategy,
            'chunker_type': type(self.chunker).__name__
        } 