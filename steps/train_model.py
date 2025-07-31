from transformers import BertForSequenceClassification
from torch.optim import AdamW
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
from typing import Any

from .utils import accuracy_fn, recall_fn, precision_fn


def train_model_step(train_dataset: Any, 
                     val_dataset: Any, 
                     tokenizer: Any, 
                     epochs: int  = 3,
                     batch_size: int = 4,
                     lr: float = 2e-5, 
                     verbose: bool = True,
                     device='cuda') -> Any:
    """
    Training model step with validation metrics

    Args:
        train_dataset: train data
        val_dataset: val data
        tokenizer: tokenizer model
        batch_size: size of batch
        lr: learning rate
        device: cpu or cuda

    Returns:
        model
    """

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2).to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_acc = 0
        train_recall = 0
        train_precision = 0
        
        print(f"Epoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            
            train_loss += loss.item()
            train_acc += accuracy_fn(batch['labels'], outputs['logits']).item()
            train_recall += recall_fn(batch['labels'], outputs['logits']).item()
            train_precision += precision_fn(batch['labels'], outputs['logits']).item()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        avg_train_loss = train_loss / len(train_loader)
        avg_train_acc = train_acc / len(train_loader)
        avg_train_recall = train_recall / len(train_loader)
        avg_train_precision = train_precision / len(train_loader)
        
        model.eval()
        val_loss = 0
        val_acc = 0
        val_recall = 0
        val_precision = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                
                val_loss += loss.item()
                val_acc += accuracy_fn(batch['labels'], outputs['logits']).item()
                val_recall += recall_fn(batch['labels'], outputs['logits']).item()
                val_precision += precision_fn(batch['labels'], outputs['logits']).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_val_acc = val_acc / len(val_loader)
        avg_val_recall = val_recall / len(val_loader)
        avg_val_precision = val_precision / len(val_loader)
        
        if verbose:
            print(f"\nEpoch {epoch + 1}")
            print(f"Train Loss: {avg_train_loss:.4f} | Train Acc: {avg_train_acc:.4f} | Train Precision: {avg_train_precision:.4f} | Train Recall: {avg_train_recall:.4f}")
            print(f"Val Loss:   {avg_val_loss:.4f} | Val Acc:   {avg_val_acc:.4f} | Val Precision:   {avg_val_precision:.4f} | Val Recall:   {avg_val_recall:.4f}")
            print("=" * 80)
            print()
    
    return model
