from typing import Any
import torch
from pathlib import Path

THRESHOLD = 0.5

def accuracy_fn(y_true: Any, y_pred: Any) -> float:
    return ((y_pred[:, 1] >= THRESHOLD).int() == y_true).int().sum() / len(y_true) 


def precision_fn(y_true: Any, y_pred: Any) -> float:
    y_pred_label = (y_pred[:, 1] >= THRESHOLD).int()
    true_positive = ((y_pred_label == 1) & (y_true == 1)).int().sum()
    predicted_positive = (y_pred_label == 1).int().sum()
    # precision: TP / (TP + FP)
    precision = true_positive / (predicted_positive + 1e-8)  
    return precision


def recall_fn(y_true: Any, y_pred: Any) -> float:
    y_pred_label = (y_pred[:, 1] >= THRESHOLD).int()
    true_positive = ((y_pred_label == 1) & (y_true == 1)).int().sum()
    actual_positive = (y_true == 1).int().sum()
    # recall: TP / (TP + FN)
    recall = true_positive / (actual_positive + 1e-8)
    return recall


def save_torch_model(model: torch.nn.Module, filename: str | Path) -> None:
    torch.save(model.state_dict(), filename)