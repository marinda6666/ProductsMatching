from torch.utils.data import DataLoader
import torch
from typing import Any

from .utils import accuracy_fn, recall_fn, precision_fn


def evaluate_step(model: Any, val_dataset: Any, tokenizer: Any, device: str = 'cuda'):
    """
    Evaluation model step

    Args:
        model: model)))
        val_dataset: validation data
        tokenizer: tokenizer model
        device: cpu or cuda
    
    Return:
        Dict: metrics
    """

    model.eval()
    val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)
    metrics = {"accuracy": 0, "precision": 0, "recall": 0, "loss": 0}
    for batch in val_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
            loss = outputs.loss
            metrics["accuracy"] += accuracy_fn(batch['labels'], outputs['logits'])
            metrics["precision"] += precision_fn(batch['labels'], outputs['logits'])
            metrics["recall"] += recall_fn(batch['labels'], outputs['logits'])
            metrics["loss"] += loss.item()
    return {k: v / len(val_loader) for k, v in metrics.items()}
