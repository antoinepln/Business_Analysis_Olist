# Business_Analysis_Olist

Analyzis a dataset provided by an e-commerce marketplace called [Olist](https://www.olist.com/), a leading e-commerce service that connects merchants to main marketplaces in Brazil. Dataset provide a wide range of offers from inventory management, dealing with reviews and customer contacts to logistic services.

The dataset consists of 9 cvs files, 100k orders from 2016 and 2018 that were made on Olist store.

Exploration of 2 specific business cases:

- The impact of removing underperforming sellers?

  **(remove\_underperforming\_sellers.ipynb)**

- The impact of restricting customer states?

  **(restricting\_customer\_states.ipynb)**

The excecutables :

data.py :

- Generate the Data, and Link the 9 tables between them according to the common keys.

Order.py :

- Features engineering, creation of Business insight related to orders.

Product.py :

- Features engineering, creation of product-related business insight

Seller.py :

- Features engineering, creation of business insight related to vendors

Utils.py :

- Function creation used in our 3 previous Python files
