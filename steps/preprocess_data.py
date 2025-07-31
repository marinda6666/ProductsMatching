from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Any, Tuple

class PairsDataset(Dataset):
    def __init__(self, df, tokenizer, max_len=128):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len


    def __len__(self):
        return len(self.df)


    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        encoded = self.tokenizer(
            row['query'],
            row['word'],
            padding='max_length',
            truncation=True,
            max_length=self.max_len,
            return_tensors='pt',
            return_overflowing_tokens=False
        )
        item = {key: val.squeeze(0) for key, val in encoded.items()}
        item['labels'] = torch.tensor(row['label'], dtype=torch.long)
        return item

def preprocess_step(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[PairsDataset, PairsDataset, Any]:
    """
    Processing data step

    Args:
        df: data
    
    Return:
        train_dataset, val_dataset, tokenizer
    """
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train_df, val_df = train_test_split(df, test_size=test_size, random_state=42)
    train_dataset = PairsDataset(train_df, tokenizer)
    val_dataset = PairsDataset(val_df, tokenizer)

    return train_dataset, val_dataset, tokenizer