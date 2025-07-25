#!/usr/bin/env python3
"""
Setup script for RAG QA System

This script initializes the project structure and creates necessary directories.
"""

import os
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories"""
    directories = [
        "data/raw",
        "data/processed", 
        "data/embeddings",
        "experiments/configs",
        "experiments/logs",
        "experiments/results",
        "notebooks",
        "tests",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")


def create_gitkeep_files():
    """Create .gitkeep files for empty directories"""
    gitkeep_dirs = [
        "data/raw",
        "data/processed",
        "data/embeddings", 
        "experiments/logs",
        "experiments/results"
    ]
    
    for directory in gitkeep_dirs:
        gitkeep_file = Path(directory) / ".gitkeep"
        gitkeep_file.touch(exist_ok=True)
        print(f"Created .gitkeep file: {gitkeep_file}")


def create_sample_config():
    """Create sample configuration file"""
    config_content = """# Sample configuration for RAG QA System

embedding:
  model_name: "BAAI/bge-small-en"
  max_length: 512
  device: "auto"
  normalize: true

llm:
  model_name: "microsoft/DialoGPT-medium"
  max_length: 2048
  temperature: 0.7
  top_p: 0.9
  device: "auto"

vector_db:
  db_type: "chroma"
  collection_name: "documents"
  persist_directory: "data/embeddings"
  distance_metric: "cosine"

chunking:
  chunk_size: 1000
  chunk_overlap: 200
  chunk_strategy: "fixed_size"

retrieval:
  top_k: 5
  similarity_threshold: 0.7
  use_reranker: false
  reranker_model: "BAAI/bge-reranker-base"

evaluation:
  metrics: ["f1", "exact_match", "bleu", "rouge"]
  test_dataset: "data/test_qa.json"
  save_results: true
  results_dir: "experiments/results"

data_dir: "data"
experiments_dir: "experiments"
log_dir: "experiments/logs"
"""
    
    config_file = Path("experiments/configs/sample.yaml")
    config_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(config_file, "w", encoding="utf-8") as f:
        f.write(config_content)
    
    print(f"Created sample config: {config_file}")


def create_test_files():
    """Create basic test files"""
    test_content = """import pytest
from src.utils.config import Config
from src.data_processing.document_loader import DocumentLoader
from src.data_processing.chunker import DocumentChunker


def test_config_loading():
    \"\"\"Test configuration loading\"\"\"
    config = Config()
    assert config.embedding.model_name is not None
    assert config.chunking.chunk_size > 0


def test_document_loader():
    \"\"\"Test document loader\"\"\"
    loader = DocumentLoader()
    supported_formats = loader.get_supported_formats()
    assert len(supported_formats) > 0


def test_chunker():
    \"\"\"Test document chunker\"\"\"
    chunker = DocumentChunker(strategy="fixed_size", chunk_size=100)
    assert chunker.strategy == "fixed_size"
"""
    
    test_file = Path("tests/test_basic.py")
    test_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(test_file, "w", encoding="utf-8") as f:
        f.write(test_content)
    
    print(f"Created test file: {test_file}")


def create_init_files():
    """Create __init__.py files for Python packages"""
    init_dirs = [
        "src",
        "src/utils",
        "src/data_processing", 
        "src/embedding",
        "src/retrieval",
        "src/generation",
        "src/evaluation",
        "tests"
    ]
    
    for directory in init_dirs:
        init_file = Path(directory) / "__init__.py"
        init_file.touch(exist_ok=True)
        print(f"Created __init__.py: {init_file}")


def main():
    """Main setup function"""
    print("Setting up RAG QA System...")
    
    # Create directories
    create_directories()
    
    # Create .gitkeep files
    create_gitkeep_files()
    
    # Create sample configuration
    create_sample_config()
    
    # Create test files
    create_test_files()
    
    # Create __init__.py files
    create_init_files()
    
    print("\nSetup completed successfully!")
    print("\nNext steps:")
    print("1. Install dependencies: pip install -r requirements.txt")
    print("2. Run basic experiment: python experiments/run_basic_rag.py")
    print("3. Run tests: pytest tests/")
    print("4. Check code quality: flake8 src/")


if __name__ == "__main__":
    main() 