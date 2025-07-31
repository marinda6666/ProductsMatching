import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from abc import ABC, abstractmethod
import joblib
from typing import List, Any
from pathlib import Path
import logging as log

MAX_LENGTH = 128
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BERT = Path('/home/marinda/Документы/ml_projects/ProductsMatching/models/checkpoints/bert_20.07_13:15.pth')


class MatchingModel(ABC):
    """Abstract class matching model"""
    @abstractmethod
    def __init__(self,
                 checkpoint: str,
                 device: str = DEVICE) -> None:
        pass
    
    @abstractmethod
    def _load_model(self, checkpoint) -> Any:
        """Loading model"""
        pass

    @abstractmethod
    def get_similarity_probs(self, text1, text2, max_length=MAX_LENGTH) -> np.ndarray:
        """Get similarity between two words"""
        pass

class Bert(MatchingModel):
    def __init__(self, 
                 model_checkpoint: str = BERT,
                 tokenizer_checkpoint: str = None,
                 device: str = DEVICE):
        self.model_checkpoint = model_checkpoint
        if tokenizer_checkpoint:
            self.tokenizer = self._load_model(tokenizer_checkpoint)
        else:
            self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.device = device
        self.model = self._load_model(model_checkpoint).to(self.device)

    def _load_model(self, checkpoint) -> Any:
        model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2, ignore_mismatched_sizes=True)
        model.load_state_dict(torch.load(checkpoint, map_location=self.device, weights_only=True))
        model.eval()
        return model
    
    def get_similarity_probs(self, text1, text2, max_length=MAX_LENGTH) -> np.ndarray:
        self.model.eval()
        inputs = self.tokenizer(
            text1.lower(),
            text2.lower(),
            padding='max_length',
            truncation=True,
            max_length=max_length,
            return_tensors='pt'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = logits.softmax(1)
        return probs.cpu().numpy()[0][1] 
