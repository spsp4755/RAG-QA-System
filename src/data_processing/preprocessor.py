"""
Text preprocessing utilities for RAG QA System
"""

import re
from typing import List, Dict, Any, Optional
from abc import ABC, abstractmethod


class BasePreprocessor(ABC):
    """Abstract base class for text preprocessors"""
    
    @abstractmethod
    def preprocess(self, text: str) -> str:
        """Preprocess text"""
        pass


class BasicTextPreprocessor(BasePreprocessor):
    """Basic text preprocessing"""
    
    def __init__(self, 
                 remove_urls: bool = True,
                 remove_emails: bool = True,
                 remove_numbers: bool = False,
                 remove_punctuation: bool = False,
                 lowercase: bool = True,
                 remove_extra_whitespace: bool = True):
        self.remove_urls = remove_urls
        self.remove_emails = remove_emails
        self.remove_numbers = remove_numbers
        self.remove_punctuation = remove_punctuation
        self.lowercase = lowercase
        self.remove_extra_whitespace = remove_extra_whitespace
    
    def preprocess(self, text: str) -> str:
        """Apply basic text preprocessing"""
        if not text:
            return ""
        
        # Remove URLs
        if self.remove_urls:
            text = re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
        
        # Remove email addresses
        if self.remove_emails:
            text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', '', text)
        
        # Remove numbers
        if self.remove_numbers:
            text = re.sub(r'\d+', '', text)
        
        # Remove punctuation
        if self.remove_punctuation:
            text = re.sub(r'[^\w\s]', '', text)
        
        # Convert to lowercase
        if self.lowercase:
            text = text.lower()
        
        # Remove extra whitespace
        if self.remove_extra_whitespace:
            text = re.sub(r'\s+', ' ', text)
            text = text.strip()
        
        return text


class MarkdownPreprocessor(BasePreprocessor):
    """Preprocessor for Markdown text"""
    
    def __init__(self, 
                 remove_headers: bool = True,
                 remove_links: bool = True,
                 remove_code_blocks: bool = True,
                 remove_inline_code: bool = False,
                 remove_lists: bool = False):
        self.remove_headers = remove_headers
        self.remove_links = remove_links
        self.remove_code_blocks = remove_code_blocks
        self.remove_inline_code = remove_inline_code
        self.remove_lists = remove_lists
    
    def preprocess(self, text: str) -> str:
        """Preprocess Markdown text"""
        if not text:
            return ""
        
        # Remove code blocks
        if self.remove_code_blocks:
            text = re.sub(r'```[\s\S]*?```', '', text)
        
        # Remove headers
        if self.remove_headers:
            text = re.sub(r'^#{1,6}\s+', '', text, flags=re.MULTILINE)
        
        # Remove links but keep link text
        if self.remove_links:
            text = re.sub(r'\[([^\]]+)\]\([^)]+\)', r'\1', text)
        
        # Remove inline code
        if self.remove_inline_code:
            text = re.sub(r'`([^`]+)`', r'\1', text)
        
        # Remove list markers
        if self.remove_lists:
            text = re.sub(r'^[\s]*[-*+]\s+', '', text, flags=re.MULTILINE)
            text = re.sub(r'^[\s]*\d+\.\s+', '', text, flags=re.MULTILINE)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text


class TextPreprocessor:
    """Main text preprocessor that combines multiple strategies"""
    
    def __init__(self, preprocessors: Optional[List[BasePreprocessor]] = None):
        """
        Initialize preprocessor with list of preprocessors
        
        Args:
            preprocessors: List of preprocessor instances to apply
        """
        if preprocessors is None:
            preprocessors = [BasicTextPreprocessor()]
        
        self.preprocessors = preprocessors
    
    def preprocess(self, text: str) -> str:
        """Apply all preprocessors to text"""
        processed_text = text
        
        for preprocessor in self.preprocessors:
            processed_text = preprocessor.preprocess(processed_text)
        
        return processed_text
    
    def preprocess_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Preprocess a single document"""
        content = document.get('content', '')
        processed_content = self.preprocess(content)
        
        processed_doc = document.copy()
        processed_doc['content'] = processed_content
        processed_doc['metadata'] = document.get('metadata', {}).copy()
        processed_doc['metadata']['preprocessed'] = True
        
        return processed_doc
    
    def preprocess_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Preprocess multiple documents"""
        processed_docs = []
        
        for doc in documents:
            processed_doc = self.preprocess_document(doc)
            processed_docs.append(processed_doc)
        
        return processed_docs
    
    def add_preprocessor(self, preprocessor: BasePreprocessor):
        """Add a new preprocessor to the pipeline"""
        self.preprocessors.append(preprocessor)
    
    def get_preprocessor_info(self) -> List[str]:
        """Get information about all preprocessors"""
        return [type(preprocessor).__name__ for preprocessor in self.preprocessors]


# Convenience functions for common preprocessing tasks
def create_basic_preprocessor(**kwargs) -> TextPreprocessor:
    """Create a basic text preprocessor"""
    return TextPreprocessor([BasicTextPreprocessor(**kwargs)])


def create_markdown_preprocessor(**kwargs) -> TextPreprocessor:
    """Create a Markdown text preprocessor"""
    return TextPreprocessor([MarkdownPreprocessor(**kwargs)])


def create_combined_preprocessor(basic_kwargs: Optional[Dict] = None, 
                               markdown_kwargs: Optional[Dict] = None) -> TextPreprocessor:
    """Create a combined preprocessor with both basic and Markdown preprocessing"""
    basic_kwargs = basic_kwargs or {}
    markdown_kwargs = markdown_kwargs or {}
    
    preprocessors = [
        MarkdownPreprocessor(**markdown_kwargs),
        BasicTextPreprocessor(**basic_kwargs)
    ]
    
    return TextPreprocessor(preprocessors) 