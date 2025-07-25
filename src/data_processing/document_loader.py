"""
Document loader for various file formats
"""

import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod

import pandas as pd


class BaseDocumentLoader(ABC):
    """Abstract base class for document loaders"""
    
    @abstractmethod
    def load(self, file_path: str) -> Dict[str, Any]:
        """Load document from file path"""
        pass
    
    @abstractmethod
    def can_load(self, file_path: str) -> bool:
        """Check if this loader can handle the file"""
        pass


class TextDocumentLoader(BaseDocumentLoader):
    """Loader for plain text files"""
    
    def can_load(self, file_path: str) -> bool:
        return file_path.lower().endswith('.txt')
    
    def load(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            'content': content,
            'metadata': {
                'source': file_path,
                'file_type': 'txt',
                'file_size': os.path.getsize(file_path)
            }
        }


class MarkdownDocumentLoader(BaseDocumentLoader):
    """Loader for Markdown files"""
    
    def can_load(self, file_path: str) -> bool:
        return file_path.lower().endswith(('.md', '.markdown'))
    
    def load(self, file_path: str) -> Dict[str, Any]:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        return {
            'content': content,
            'metadata': {
                'source': file_path,
                'file_type': 'markdown',
                'file_size': os.path.getsize(file_path)
            }
        }


class CSVDocumentLoader(BaseDocumentLoader):
    """Loader for CSV files"""
    
    def can_load(self, file_path: str) -> bool:
        return file_path.lower().endswith('.csv')
    
    def load(self, file_path: str) -> Dict[str, Any]:
        df = pd.read_csv(file_path)
        
        # Convert DataFrame to text representation
        content = df.to_string(index=False)
        
        return {
            'content': content,
            'metadata': {
                'source': file_path,
                'file_type': 'csv',
                'file_size': os.path.getsize(file_path),
                'rows': len(df),
                'columns': list(df.columns)
            }
        }


class DocumentLoader:
    """Main document loader that handles multiple file formats"""
    
    def __init__(self):
        self.loaders = [
            TextDocumentLoader(),
            MarkdownDocumentLoader(),
            CSVDocumentLoader(),
        ]
    
    def load_document(self, file_path: str) -> Dict[str, Any]:
        """Load a single document"""
        file_path = str(file_path)
        
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")
        
        # Find appropriate loader
        for loader in self.loaders:
            if loader.can_load(file_path):
                return loader.load(file_path)
        
        raise ValueError(f"No loader found for file: {file_path}")
    
    def load_directory(self, directory_path: str, recursive: bool = True) -> List[Dict[str, Any]]:
        """Load all documents from a directory"""
        directory_path = Path(directory_path)
        documents = []
        
        if recursive:
            file_paths = directory_path.rglob('*')
        else:
            file_paths = directory_path.glob('*')
        
        for file_path in file_paths:
            if file_path.is_file():
                try:
                    doc = self.load_document(str(file_path))
                    documents.append(doc)
                except (ValueError, FileNotFoundError) as e:
                    print(f"Warning: Could not load {file_path}: {e}")
        
        return documents
    
    def load_multiple_files(self, file_paths: List[str]) -> List[Dict[str, Any]]:
        """Load multiple documents from file paths"""
        documents = []
        
        for file_path in file_paths:
            try:
                doc = self.load_document(file_path)
                documents.append(doc)
            except (ValueError, FileNotFoundError) as e:
                print(f"Warning: Could not load {file_path}: {e}")
        
        return documents
    
    def get_supported_formats(self) -> List[str]:
        """Get list of supported file formats"""
        formats = []
        for loader in self.loaders:
            if hasattr(loader, 'can_load'):
                # This is a simple way to get supported formats
                # In a real implementation, you might want to store this explicitly
                formats.append(type(loader).__name__.replace('DocumentLoader', '').lower())
        return formats 