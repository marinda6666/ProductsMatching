# ProductsMatching
**ℹ️ Info**


Service for matching electronics products from ebay. 

It takes a CSV file with products and prices and collects statistics for every item (mean price on eBay, max price, min price, etc.)

<div align="center"> <img width="500" height="425" alt="image" src="https://github.com/user-attachments/assets/730905ea-9c26-4b5d-a8fb-29df53a007fe" /> </div>
At first glance, it seems that eBay's search engine should compare products well on its own, but here's an example for the popular query - iPhone 16 256 GB.

<div align="center"> <img width="500" alt="eBay search results showing poor matching" src="https://github.com/user-attachments/assets/8b2349f0-52bb-4c4b-bf95-60c7a7d5af66" /> </div>
As you can see, only one item satisfies our requirements.

I used the eBay Developer API for finding items and then compared every product with my query using BERT. The model has been trained on my own CSV dataset containing 10,000 rows.

🛠 Microservice Architecture

<img width="1384" height="345" alt="image" src="https://github.com/user-attachments/assets/1d7118f7-b17b-4750-b7b3-79113c77b092" />

* *process file* - take csv file and find items with EBay API, then compare every query-word pair via match items service
* *match items* - connect to triton and sent it tokenizer pair
*  *triton* - contain BERT which process input and return probabilities

  
**🚀 Run**

for running service you need get EBay Api token - https://developer.ebay.com/my/api_test_tool?index=0&env=production

```
cd src
docker-compose build && docker-compose up -d
```
