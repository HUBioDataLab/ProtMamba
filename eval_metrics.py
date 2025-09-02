from transformers.trainer_utils import EvalPrediction
from sklearn.metrics import precision_recall_fscore_support, accuracy_score
import numpy as np
import torch

def compute_metrics(eval_pred: EvalPrediction):
    logits, labels = eval_pred
    logits = logits[:, :-1]
    labels = labels[:, 1:]
    preds = np.argmax(logits, axis=-1)

    # Create a mask for valid tokens
    valid_mask = (labels != -100) & (labels != 0)

    # Apply mask to labels, logits, and predictions
    labels = labels[valid_mask]
    logits = logits[valid_mask]
    preds = preds[valid_mask]

    # Convert to PyTorch tensors
    logits = torch.tensor(logits)
    labels = torch.tensor(labels)

    # Compute loss and perplexity
    loss = torch.nn.functional.cross_entropy(logits, labels, reduction='mean')
    perplexity = torch.exp(loss).item()

    # Compute other metrics
    accuracy = accuracy_score(labels, preds)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true=labels, 
        y_pred=preds, 
        average='macro', 
        zero_division=0
    )
    
    precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true=labels, 
        y_pred=preds, 
        average='micro', 
        zero_division=0
    )
    
    return {
        "accuracy"        : accuracy,
        "precision_micro" : precision_micro,
        "recall_micro"    : recall_micro,
        "f1_micro"        : f1_micro,
        "precision_macro" : precision_macro,
        "recall_macro"    : recall_macro,
        "f1_macro"        : f1_macro,
        "perplexity"      : perplexity,
        'loss'            : loss
    }
