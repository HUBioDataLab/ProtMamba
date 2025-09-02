"""
Metrics computation module for model evaluation.
"""
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support
)
from transformers.trainer_utils import EvalPrediction


def compute_metrics(eval_pred: EvalPrediction, tokenizer=None):
    """
    Compute evaluation metrics for language modeling.
    
    Args:
        eval_pred: EvalPrediction object containing predictions and labels
        tokenizer: Tokenizer for identifying special tokens
        
    Returns:
        Dictionary containing computed metrics
    """
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    
    # Shift for causal LM - predict next token
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    
    # Get predictions
    preds = np.argmax(logits, axis=-1)
    
    # Create mask for valid tokens (exclude padding and special tokens)
    if tokenizer is not None:
        valid_mask = (labels != -100) & (labels != tokenizer.pad_token_id)
    else:
        valid_mask = labels != -100
    
    # Apply mask to labels, logits, and predictions
    labels_masked = labels[valid_mask]
    logits_masked = logits[valid_mask]
    preds_masked = preds[valid_mask]
    
    # Convert to PyTorch tensors for loss computation
    logits_tensor = torch.tensor(logits_masked)
    labels_tensor = torch.tensor(labels_masked)
    
    # Compute loss and perplexity
    loss = torch.nn.functional.cross_entropy(
        logits_tensor, 
        labels_tensor, 
        reduction='mean'
    )
    
    try:
        perplexity = torch.exp(loss).item()
    except OverflowError:
        perplexity = float('inf')
    
    # Compute accuracy
    accuracy = accuracy_score(labels_masked, preds_masked)
    
    # Compute precision, recall, and F1 scores
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true=labels_masked,
        y_pred=preds_masked,
        average='macro',
        zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true=labels_masked,
        y_pred=preds_masked,
        average='micro',
        zero_division=0
    )
    
    # Compute top-k accuracy
    top5_acc = compute_topk_accuracy(logits_masked, labels_masked, k=5)
    top10_acc = compute_topk_accuracy(logits_masked, labels_masked, k=10)
    
    return {
        "accuracy": accuracy,
        "top5_accuracy": top5_acc,
        "top10_accuracy": top10_acc,
        "precision_micro": precision_micro,
        "recall_micro": recall_micro,
        "f1_micro": f1_micro,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "perplexity": perplexity,
        "loss": loss.item()
    }


def compute_topk_accuracy(logits, labels, k=5):
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model predictions
        labels: True labels
        k: Number of top predictions to consider
        
    Returns:
        Top-k accuracy score
    """
    # Get top k predictions
    topk_preds = np.argsort(logits, axis=-1)[:, -k:]
    
    # Check if true label is in top k
    correct = np.any(topk_preds == labels[:, np.newaxis], axis=-1)
    
    return np.mean(correct)


def create_compute_metrics_fn(tokenizer):
    """
    Create a compute_metrics function with tokenizer bound.
    
    Args:
        tokenizer: Tokenizer to use for metric computation
        
    Returns:
        Function for computing metrics
    """
    def compute_metrics_wrapper(eval_pred):
        return compute_metrics(eval_pred, tokenizer)
    
    return compute_metrics_wrapper