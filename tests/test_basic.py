import pytest
from src.utils.config import Config
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.chunker import DocumentChunker


def test_config_loading():
    """Test configuration loading"""
    config = Config()
    assert config.embedding.model_name is not None
    assert config.chunking.chunk_size > 0


def test_document_loader():
    """Test document loader"""
    loader = DocumentLoader()
    supported_formats = loader.get_supported_formats()
    assert len(supported_formats) > 0


def test_chunker():
    """Test document chunker"""
    chunker = DocumentChunker(strategy="fixed_size", chunk_size=100)
    assert chunker.strategy == "fixed_size"
