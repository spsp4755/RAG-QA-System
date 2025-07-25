#!/usr/bin/env python3
"""
Basic RAG QA System Experiment

This script demonstrates a basic RAG pipeline with:
1. Document loading and preprocessing
2. Document chunking
3. Embedding generation
4. Vector storage
5. Query processing and answer generation
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils.config import Config
from utils.logger import setup_logger, ExperimentLogger
from data_processing.document_loader import DocumentLoader
from data_processing.chunker import DocumentChunker
from data_processing.preprocessor import TextPreprocessor


def create_sample_data():
    """Create sample documents for testing"""
    sample_docs = [
        {
            "filename": "sample1.txt",
            "content": """
            인공지능(AI)은 인간의 학습능력과 추론능력, 지각능력, 자연언어의 이해능력 등을 
            컴퓨터 프로그램으로 실현한 기술입니다. 머신러닝은 AI의 한 분야로, 데이터로부터 
            학습하여 패턴을 찾고 예측을 수행하는 기술입니다. 딥러닝은 머신러닝의 하위 분야로, 
            인공신경망을 사용하여 복잡한 패턴을 학습합니다.
            """
        },
        {
            "filename": "sample2.txt", 
            "content": """
            자연어처리(NLP)는 인간의 언어를 컴퓨터가 이해하고 처리할 수 있도록 하는 
            기술입니다. 텍스트 분류, 감정 분석, 기계번역, 질의응답 등 다양한 분야에서 
            활용됩니다. 최근에는 트랜스포머 모델과 BERT, GPT 등의 사전학습 모델이 
            자연어처리 성능을 크게 향상시켰습니다.
            """
        },
        {
            "filename": "sample3.txt",
            "content": """
            RAG(Retrieval-Augmented Generation)는 검색과 생성을 결합한 기술로, 
            외부 지식베이스에서 관련 정보를 검색하여 더 정확하고 신뢰할 수 있는 
            답변을 생성합니다. 이는 대규모 언어모델의 환각(hallucination) 문제를 
            해결하는 효과적인 방법 중 하나입니다.
            """
        }
    ]
    
    # Create data directory
    data_dir = Path("data/raw")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Write sample documents
    for doc in sample_docs:
        file_path = data_dir / doc["filename"]
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(doc["content"].strip())
    
    return [str(data_dir / doc["filename"]) for doc in sample_docs]


def run_basic_rag_experiment():
    """Run basic RAG experiment"""
    
    # Setup configuration
    config = Config()
    experiment_name = config.get_experiment_name()
    
    # Setup logging
    with ExperimentLogger(experiment_name) as logger:
        logger.info("Starting Basic RAG Experiment")
        logger.info(f"Configuration: {config.__dict__}")
        
        # Step 1: Create sample data
        logger.info("Step 1: Creating sample data")
        sample_files = create_sample_data()
        logger.info(f"Created {len(sample_files)} sample files")
        
        # Step 2: Load documents
        logger.info("Step 2: Loading documents")
        loader = DocumentLoader()
        documents = loader.load_multiple_files(sample_files)
        logger.info(f"Loaded {len(documents)} documents")
        
        # Step 3: Preprocess documents
        logger.info("Step 3: Preprocessing documents")
        preprocessor = TextPreprocessor()
        processed_docs = preprocessor.preprocess_documents(documents)
        logger.info(f"Preprocessed {len(processed_docs)} documents")
        
        # Step 4: Chunk documents
        logger.info("Step 4: Chunking documents")
        chunker = DocumentChunker(
            strategy=config.chunking.chunk_strategy,
            chunk_size=config.chunking.chunk_size,
            chunk_overlap=config.chunking.chunk_overlap
        )
        chunks = chunker.chunk_documents(processed_docs)
        logger.info(f"Created {len(chunks)} chunks")
        
        # Step 5: Display chunk statistics
        logger.info("Step 5: Chunk statistics")
        chunk_sizes = [len(chunk['content']) for chunk in chunks]
        logger.info(f"Average chunk size: {sum(chunk_sizes) / len(chunk_sizes):.2f} characters")
        logger.info(f"Min chunk size: {min(chunk_sizes)} characters")
        logger.info(f"Max chunk size: {max(chunk_sizes)} characters")
        
        # Step 6: Save processed data
        logger.info("Step 6: Saving processed data")
        processed_dir = Path("data/processed")
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save chunks
        chunks_file = processed_dir / "chunks.json"
        with open(chunks_file, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved chunks to {chunks_file}")
        
        # Step 7: Test query processing (simplified)
        logger.info("Step 7: Testing query processing")
        test_queries = [
            "인공지능이란 무엇인가요?",
            "자연어처리의 활용 분야는?",
            "RAG 기술의 장점은?"
        ]
        
        for query in test_queries:
            logger.info(f"Query: {query}")
            # Simple keyword matching for demonstration
            relevant_chunks = []
            for chunk in chunks:
                if any(keyword in chunk['content'] for keyword in query.split()):
                    relevant_chunks.append(chunk)
            
            logger.info(f"Found {len(relevant_chunks)} relevant chunks")
            if relevant_chunks:
                logger.info(f"Top chunk: {relevant_chunks[0]['content'][:100]}...")
        
        # Step 8: Save experiment results
        logger.info("Step 8: Saving experiment results")
        results = {
            "experiment_name": experiment_name,
            "timestamp": datetime.now().isoformat(),
            "config": {
                "chunking_strategy": config.chunking.chunk_strategy,
                "chunk_size": config.chunking.chunk_size,
                "chunk_overlap": config.chunking.chunk_overlap,
                "num_documents": len(documents),
                "num_chunks": len(chunks),
                "avg_chunk_size": sum(chunk_sizes) / len(chunk_sizes)
            },
            "test_queries": test_queries
        }
        
        results_dir = Path("experiments/results")
        results_dir.mkdir(parents=True, exist_ok=True)
        
        results_file = results_dir / f"{experiment_name}_results.json"
        with open(results_file, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Saved results to {results_file}")
        
        logger.info("Basic RAG experiment completed successfully!")


if __name__ == "__main__":
    run_basic_rag_experiment() 