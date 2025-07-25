"""
Configuration management for RAG QA System
"""

import os
import yaml
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
from pathlib import Path


@dataclass
class EmbeddingConfig:
    """Embedding model configuration"""
    model_name: str = "BAAI/bge-small-en"
    model_path: Optional[str] = None
    max_length: int = 512
    device: str = "auto"
    normalize: bool = True


@dataclass
class LLMConfig:
    """LLM configuration"""
    model_name: str = "microsoft/DialoGPT-medium"
    model_path: Optional[str] = None
    max_length: int = 2048
    temperature: float = 0.7
    top_p: float = 0.9
    device: str = "auto"


@dataclass
class VectorDBConfig:
    """Vector database configuration"""
    db_type: str = "chroma"  # "chroma" or "faiss"
    collection_name: str = "documents"
    persist_directory: str = "data/embeddings"
    distance_metric: str = "cosine"


@dataclass
class ChunkingConfig:
    """Document chunking configuration"""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    chunk_strategy: str = "fixed_size"  # "fixed_size", "paragraph", "semantic"


@dataclass
class RetrievalConfig:
    """Retrieval configuration"""
    top_k: int = 5
    similarity_threshold: float = 0.7
    use_reranker: bool = False
    reranker_model: str = "BAAI/bge-reranker-base"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    metrics: Optional[List[str]] = None
    test_dataset: str = "data/test_qa.json"
    save_results: bool = True
    results_dir: str = "experiments/results"
    
    def __post_init__(self):
        if self.metrics is None:
            self.metrics = ['f1', 'exact_match', 'bleu', 'rouge']


class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "experiments/configs/default.yaml"
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        if os.path.exists(self.config_path):
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_data = yaml.safe_load(f)
        else:
            config_data = self._get_default_config()
            self._save_config(config_data)
        
        # Set attributes from config data
        self.embedding = EmbeddingConfig(**config_data.get('embedding', {}))
        self.llm = LLMConfig(**config_data.get('llm', {}))
        self.vector_db = VectorDBConfig(**config_data.get('vector_db', {}))
        self.chunking = ChunkingConfig(**config_data.get('chunking', {}))
        self.retrieval = RetrievalConfig(**config_data.get('retrieval', {}))
        self.evaluation = EvaluationConfig(**config_data.get('evaluation', {}))
        
        # General settings
        self.data_dir = config_data.get('data_dir', 'data')
        self.experiments_dir = config_data.get('experiments_dir', 'experiments')
        self.log_dir = config_data.get('log_dir', 'experiments/logs')
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            'embedding': {
                'model_name': 'BAAI/bge-small-en',
                'max_length': 512,
                'device': 'auto',
                'normalize': True
            },
            'llm': {
                'model_name': 'microsoft/DialoGPT-medium',
                'max_length': 2048,
                'temperature': 0.7,
                'top_p': 0.9,
                'device': 'auto'
            },
            'vector_db': {
                'db_type': 'chroma',
                'collection_name': 'documents',
                'persist_directory': 'data/embeddings',
                'distance_metric': 'cosine'
            },
            'chunking': {
                'chunk_size': 1000,
                'chunk_overlap': 200,
                'chunk_strategy': 'fixed_size'
            },
            'retrieval': {
                'top_k': 5,
                'similarity_threshold': 0.7,
                'use_reranker': False,
                'reranker_model': 'BAAI/bge-reranker-base'
            },
            'evaluation': {
                'metrics': ['f1', 'exact_match', 'bleu', 'rouge'],
                'test_dataset': 'data/test_qa.json',
                'save_results': True,
                'results_dir': 'experiments/results'
            },
            'data_dir': 'data',
            'experiments_dir': 'experiments',
            'log_dir': 'experiments/logs'
        }
    
    def _save_config(self, config_data: Dict[str, Any]):
        """Save configuration to file"""
        os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
        with open(self.config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, default_flow_style=False, allow_unicode=True)
    
    def save(self):
        """Save current configuration"""
        config_data = {
            'embedding': self.embedding.__dict__,
            'llm': self.llm.__dict__,
            'vector_db': self.vector_db.__dict__,
            'chunking': self.chunking.__dict__,
            'retrieval': self.retrieval.__dict__,
            'evaluation': self.evaluation.__dict__,
            'data_dir': self.data_dir,
            'experiments_dir': self.experiments_dir,
            'log_dir': self.log_dir
        }
        self._save_config(config_data)
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                if isinstance(value, dict) and hasattr(getattr(self, key), '__dict__'):
                    # Update nested config object
                    for k, v in value.items():
                        if hasattr(getattr(self, key), k):
                            setattr(getattr(self, key), k, v)
                else:
                    setattr(self, key, value)
    
    def get_experiment_name(self) -> str:
        """Generate experiment name based on current config"""
        return f"rag_{self.embedding.model_name.split('/')[-1]}_{self.llm.model_name.split('/')[-1]}" 