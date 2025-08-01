# ProductsMatching
**ℹ️ Info**


Service for matching electronics products from ebay. 

It takes a CSV file with products and prices and collects statistics for every item (mean price on eBay, max price, min price, etc.)

<div align="center"> <img width="400" height="325" alt="ProductsMatching interface" src="https://github.com/user-attachments/assets/9893a292-07a6-4705-9280-fcc7e24bdbcc" /> </div>
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
```
cd src
docker-compose build && docker-compose up -d
```
