import torch
import numpy as np
import pandas as pd
from typing import List, Any
import logging as log
import os
import sys
from dotenv import load_dotenv
import requests

from searchers import EbaySearcher

# root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
# if root_dir not in sys.path:
#     sys.path.append(root_dir)

# from models.matching_models import Bert


load_dotenv()
EBAY_TOKEN = os.getenv("EBAY_TOKEN")
THRESHOLD = 0.7

def calculate_item_stats(query: str, items: List[str|int]):
    prices = []
    print('-----------------------------------------------------')
    print(f'For {query}:')
    ind = 0
    for name, price in items:
        url = "http://127.0.0.1:8090/predict/"
        params = {
            "text1": query,
            "text2": name
        }

        response = requests.post(url, params=params)
        prob = response.json()[1]
        if prob >= THRESHOLD:
            print(f'{ind}. {name} - {price} ({prob:.4f})')
            ind += 1
            prices.append(price)
        else:
            print(f'NOTMACH: {name} - {price} ({prob:.4f})')
    print('-----------------------------------------------------')
    if not prices:
        return {"mean": 0, "std": 0, "min": 0, "max": 0}
    prices = list(map(float, prices))
    return {
        "mean": round(np.mean(prices), 2),
        "std": round(np.std(prices), 2),
        "min": round(np.min(prices), 2),
        "max": round(np.max(prices), 2)
    }
    
    

def process_table(df: pd.DataFrame):
    searcher = EbaySearcher(EBAY_TOKEN)
    for index, row in df.iterrows():
        finded_items = searcher.search(row['product_name'])
        stats = calculate_item_stats(row['product_name'], finded_items)
        df.loc[index, stats.keys()] = list(stats.values())
    return df


