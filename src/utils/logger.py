"""
Logging utilities for RAG QA System
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

from loguru import logger


def setup_logger(
    log_dir: str = "experiments/logs",
    level: str = "INFO",
    rotation: str = "1 day",
    retention: str = "30 days",
    experiment_name: Optional[str] = None
) -> logger:
    """
    Setup logger for the RAG QA System
    
    Args:
        log_dir: Directory to store log files
        level: Logging level
        rotation: Log rotation policy
        retention: Log retention policy
        experiment_name: Name of the current experiment
    
    Returns:
        Configured logger instance
    """
    # Create log directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Remove default handler
    logger.remove()
    
    # Add console handler
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=level,
        colorize=True
    )
    
    # Generate log filename
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if experiment_name:
        log_filename = f"{experiment_name}_{timestamp}.log"
    else:
        log_filename = f"rag_qa_{timestamp}.log"
    
    log_path = Path(log_dir) / log_filename
    
    # Add file handler
    logger.add(
        log_path,
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level=level,
        rotation=rotation,
        retention=retention,
        compression="zip"
    )
    
    logger.info(f"Logger initialized. Log file: {log_path}")
    return logger


def get_experiment_logger(experiment_name: str, config: Optional[dict] = None) -> logger:
    """
    Get a logger specifically for an experiment
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration
    
    Returns:
        Configured logger for the experiment
    """
    logger = setup_logger(experiment_name=experiment_name)
    
    if config:
        logger.info(f"Starting experiment: {experiment_name}")
        logger.info(f"Configuration: {config}")
    
    return logger


class ExperimentLogger:
    """Context manager for experiment logging"""
    
    def __init__(self, experiment_name: str, log_dir: str = "experiments/logs"):
        self.experiment_name = experiment_name
        self.log_dir = log_dir
        self.logger = None
    
    def __enter__(self):
        self.logger = setup_logger(
            log_dir=self.log_dir,
            experiment_name=self.experiment_name
        )
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        return self.logger
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type and self.logger:
            self.logger.error(f"Experiment failed: {exc_val}")
        elif self.logger:
            self.logger.info(f"Experiment completed: {self.experiment_name}")
        return False  # Don't suppress exceptions 