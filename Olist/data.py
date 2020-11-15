import os
import pandas as pd


class Olist:

    def get_data(self):
        """
        01-01 > This function returns all Olist datasets
        as DataFrames within a Python dict.
        """
        # Hint: You will need to find the absolute path of the csv folder in order to call this method from anywhere.
        # Hint 2: look at python __file__ attribute
        csv_path = "/Users/antoine/code/antoinepln/data-challenges/04-Decision-Science/data/csv"
        file_names_list = os.listdir(csv_path)
        del file_names_list[4]
        def key_from_file_name(f):
            a = f.replace(".csv","").replace("olist_","").replace("_dataset","")
            return a
        data = {}
        for f in file_names_list :
            path = csv_path + "/" + f
            data[key_from_file_name(f)]=pd.read_csv(path)
        return data


    def get_matching_table(self):
        """
        01-01 > This function returns a matching table between
        columns [ "order_id", "review_id", "customer_id", "product_id", "seller_id"]
        """
        data = self.get_data()
        orders = data['orders']
        sellers = data['sellers']
        products = data['products']
        items = data['order_items']
        reviews = data['order_reviews']
        matching_table = orders.merge(reviews, on ='order_id', how ='outer')
        matching_table = matching_table.merge(items, on='order_id',how='outer')[['order_id','review_id','customer_id','product_id','seller_id']]
        return matching_table

