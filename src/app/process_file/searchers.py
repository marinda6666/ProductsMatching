import requests
from typing import List

class EbaySearcher:
    def __init__(self, token: str) -> None:
        self.token = token

    def search(self, item: str) -> List[List[str|int]]:
        """
        Search item on Ebay and return list of close items with name and price

        Args:
            item: product name
        
        """
        url = f'https://api.ebay.com/buy/browse/v1/item_summary/search?q={item}'
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json"
        }

        params = {
            "q": item,
            "limit": 10,  
            "buyingOptions": "FIXED_PRICE"  
        }

        response = requests.get(url, headers=headers, params=params)
        res = []
        if response.status_code == 200:
            items = response.json().get('itemSummaries', [])
            res = []
            for i, item_info in enumerate(items):
                # print(f"{i}. {item_info.get('title')} - {item_info.get('price')['value']}")
                res.append([item_info.get('title'), item_info.get('price')['value']])
        else:
            print("ERROR WITH EBAY SEARCHER:", response.text)
        return res