"""
Evaluation metrics for RAG QA System
"""

import re
from typing import List, Dict, Any, Optional
from difflib import SequenceMatcher

import numpy as np
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def normalize_text(text: str) -> str:
    """Normalize text for evaluation"""
    # Convert to lowercase
    text = text.lower()
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove punctuation (optional)
    # text = re.sub(r'[^\w\s]', '', text)
    return text.strip()


def exact_match(prediction: str, ground_truth: str) -> float:
    """Calculate exact match score"""
    pred_norm = normalize_text(prediction)
    gt_norm = normalize_text(ground_truth)
    return 1.0 if pred_norm == gt_norm else 0.0


def f1_score(prediction: str, ground_truth: str) -> float:
    """Calculate F1 score for text similarity"""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens or not gt_tokens:
        return 0.0
    
    # Calculate common tokens
    common = set(pred_tokens) & set(gt_tokens)
    
    if not common:
        return 0.0
    
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * (precision * recall) / (precision + recall)


def bleu_score(prediction: str, ground_truth: str) -> float:
    """Calculate BLEU score"""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    
    if not pred_tokens:
        return 0.0
    
    # Use smoothing function for short sequences
    smoothing = SmoothingFunction().method1
    
    try:
        score = sentence_bleu([gt_tokens], pred_tokens, smoothing_function=smoothing)
        return score
    except:
        return 0.0


def rouge_score(prediction: str, ground_truth: str) -> Dict[str, float]:
    """Calculate ROUGE scores"""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(ground_truth, prediction)
    
    return {
        'rouge1': scores['rouge1'].fmeasure,
        'rouge2': scores['rouge2'].fmeasure,
        'rougeL': scores['rougeL'].fmeasure
    }


def semantic_similarity(prediction: str, ground_truth: str) -> float:
    """Calculate semantic similarity using sequence matcher"""
    return SequenceMatcher(None, prediction, ground_truth).ratio()


def calculate_metrics(
    predictions: List[str],
    ground_truths: List[str],
    metrics: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Calculate multiple evaluation metrics
    
    Args:
        predictions: List of predicted answers
        ground_truths: List of ground truth answers
        metrics: List of metrics to calculate
    
    Returns:
        Dictionary of metric scores
    """
    if metrics is None:
        metrics = ['exact_match', 'f1', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'semantic_similarity']
    
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground truths must have the same length")
    
    results = {}
    
    for metric in metrics:
        if metric == 'exact_match':
            scores = [exact_match(pred, gt) for pred, gt in zip(predictions, ground_truths)]
            results[metric] = np.mean(scores)
        
        elif metric == 'f1':
            scores = [f1_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
            results[metric] = np.mean(scores)
        
        elif metric == 'bleu':
            scores = [bleu_score(pred, gt) for pred, gt in zip(predictions, ground_truths)]
            results[metric] = np.mean(scores)
        
        elif metric.startswith('rouge'):
            rouge_scores = []
            for pred, gt in zip(predictions, ground_truths):
                rouge_result = rouge_score(pred, gt)
                rouge_scores.append(rouge_result[metric])
            results[metric] = np.mean(rouge_scores)
        
        elif metric == 'semantic_similarity':
            scores = [semantic_similarity(pred, gt) for pred, gt in zip(predictions, ground_truths)]
            results[metric] = np.mean(scores)
    
    return results


def calculate_retrieval_metrics(
    retrieved_docs: List[List[str]],
    relevant_docs: List[List[str]],
    k: int = 5
) -> Dict[str, float]:
    """
    Calculate retrieval metrics
    
    Args:
        retrieved_docs: List of retrieved document lists for each query
        relevant_docs: List of relevant document lists for each query
        k: Number of top documents to consider
    
    Returns:
        Dictionary of retrieval metrics
    """
    precisions = []
    recalls = []
    f1_scores = []
    
    for retrieved, relevant in zip(retrieved_docs, relevant_docs):
        # Consider only top-k retrieved documents
        retrieved_k = retrieved[:k]
        
        # Calculate intersection
        intersection = set(retrieved_k) & set(relevant)
        
        # Calculate precision and recall
        precision = len(intersection) / len(retrieved_k) if retrieved_k else 0
        recall = len(intersection) / len(relevant) if relevant else 0
        
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    return {
        f'precision@{k}': np.mean(precisions),
        f'recall@{k}': np.mean(recalls),
        f'f1@{k}': np.mean(f1_scores)
    }


def format_metrics(metrics: Dict[str, float], decimal_places: int = 4) -> str:
    """Format metrics for display"""
    formatted = []
    for metric, value in metrics.items():
        formatted.append(f"{metric}: {value:.{decimal_places}f}")
    return " | ".join(formatted) 