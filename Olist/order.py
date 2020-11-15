import os
import pandas as pd
import numpy as np
from olist.utils import haversine_distance, dim_five_star, dim_one_star,retard
from olist.data import Olist


class Order:
    '''
    DataFrames containing all orders delivered as index, and various properties of these orders as columns
    '''

    def __init__(self):
        self.data = Olist().get_data()

    def get_wait_time(self, is_delivered=True):
        """
        02-01 > Returns a DataFrame with:
        [order_id, wait_time, expected_wait_time ,delay_vs_expected]
        and filtering out non-delivered orders unless specified
        """
        self.orders = self.data['orders']
        self.orders['order_purchase_timestamp'] = pd.to_datetime(self.orders['order_purchase_timestamp'])
        self.orders['order_delivered_customer_date'] = pd.to_datetime(self.orders['order_delivered_customer_date'])
        self.orders['order_estimated_delivery_date'] = pd.to_datetime(self.orders['order_estimated_delivery_date'])
        delivery_day = self.orders['order_delivered_customer_date']
        delivery_expected = self.orders['order_estimated_delivery_date']
        order_day = self.orders['order_purchase_timestamp']
        wait_time = delivery_day - order_day
        wait_time = wait_time.map(lambda x : x.days)
        expected_wait_time = delivery_expected - order_day
        expected_wait_time = expected_wait_time.map(lambda x : x.days)
        delay_vs_expected = (delivery_day - delivery_expected).map(lambda x : x.days)
        delay_vs_expected = delay_vs_expected.apply(lambda x: retard(x))
        self.orders['wait_time']= wait_time
        self.orders['expected_wait_time'] = expected_wait_time
        self.orders['delay_vs_expected'] = delay_vs_expected
        get_wait_time_df = self.orders[['order_id','wait_time','expected_wait_time','delay_vs_expected']]
        get_wait_time_df.set_index('order_id', inplace = True)
        return get_wait_time_df

    def get_review_score(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, dim_is_five_star, dim_is_one_star
        """
        self.reviews = self.data['order_reviews']
        self.reviews["dim_is_five_star"] = self.reviews["review_score"].map(dim_five_star)
        self.reviews["dim_is_one_star"] = self.reviews["review_score"].map(dim_one_star)
        get_review_score_df = self.reviews[['order_id','dim_is_five_star','dim_is_one_star','review_score']]
        get_review_score_df.set_index('order_id', inplace = True)
        return get_review_score_df
    def get_number_products(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_products
        """
        self.items = self.data['order_items']
        get_number_products_df = self.items.groupby(by = 'order_id').nunique()[['product_id']]
        get_number_products_df.columns = ['number_of_products']
        return get_number_products_df

    def get_number_sellers(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, number_of_sellers
        """
        self.items = self.data['order_items']
        get_number_sellers_df = self.items.groupby(by = 'order_id').nunique()[['seller_id']]
        get_number_sellers_df.columns = ['number_of_sellers']
        return get_number_sellers_df

    def get_price_and_freight(self):
        """
        02-01 > Returns a DataFrame with:
        order_id, price, freight_value
        """
        self.items = self.data['order_items']
        get_price_and_freight_df = self.items[['order_id', 'price', 'freight_value']]
        get_price_and_freight_df.set_index('order_id', inplace = True)
        return get_price_and_freight_df

    def get_distance_seller_customer(self):
        """
        02-01 > Returns a DataFrame with order_id
        and distance between seller and customer
        """
        # Optional
        # Hint: you can use the haversine_distance logic coded in olist/utils.py
        data = self.data
        matching_table = Olist().get_matching_table()

        # Since one zipcode can map to multiple (lat, lng), take first one
        geo = data['geolocation']
        geo = geo.groupby('geolocation_zip_code_prefix',
                          as_index=False).first()

        # Select sellers and customers
        sellers = data['sellers']
        customers = data['customers']

        # Merge geo_location for sellers
        sellers_mask_columns = ['seller_id', 'seller_zip_code_prefix',
                                'seller_city', 'seller_state',
                                'geolocation_lat', 'geolocation_lng']

        sellers_geo = sellers.merge(geo,
                                    how='left',
                                    left_on='seller_zip_code_prefix',
                                    right_on='geolocation_zip_code_prefix')[sellers_mask_columns]

        # Merge geo_location for customers
        customers_mask_columns = ['customer_id', 'customer_zip_code_prefix',
                                  'customer_city', 'customer_state',
                                  'geolocation_lat', 'geolocation_lng']

        customers_geo = customers.merge(geo,
                                        how='left',
                                        left_on='customer_zip_code_prefix',
                                        right_on='geolocation_zip_code_prefix')[customers_mask_columns]

        # Use the matching table and merge customers and sellers
        matching_geo = matching_table.merge(sellers_geo,
                                            on='seller_id')\
                                     .merge(customers_geo,
                                            on='customer_id',
                                            suffixes=('_seller',
                                                      '_customer'))
        # Remove na()
        matching_geo = matching_geo.dropna()

        matching_geo.loc[:, 'distance_seller_customer'] =\
            matching_geo.apply(lambda row:
                               haversine_distance(row['geolocation_lng_seller'],
                                                  row['geolocation_lat_seller'],
                                                  row['geolocation_lng_customer'],
                                                  row['geolocation_lat_customer']),
                               axis=1)
        # Since an order can have multiple sellers,
        # return the average of the distance per order
        order_distance =\
            matching_geo.groupby('order_id',
                                 as_index=False).agg({'distance_seller_customer':
                                                      'mean'})

        return order_distance
    def get_training_data(self, is_delivered=True):
        """
        02-01 > Returns a clean DataFrame (without NaN), with the following columns:
        [order_id, wait_time, delay_vs_expected,
        dim_is_five_star, dim_is_one_star, review_score, number_of_products,
        number_of_sellers, freight_value, distance_customer_seller]
        """
        wait_time_df = self.get_wait_time()
        review_score_df = self.get_review_score()
        number_products_df = self.get_number_products()
        number_sellers_df = self.get_number_sellers()
        price_and_freight_df = self.get_price_and_freight()
        get_distance_seller_customer_df = self.get_distance_seller_customer()
        get_training_data_df = wait_time_df.merge(review_score_df, on = 'order_id')
        get_training_data_df = get_training_data_df.merge(number_products_df, on='order_id')
        get_training_data_df = get_training_data_df.merge(price_and_freight_df, on ='order_id')
        get_training_data_df = get_training_data_df.merge(get_distance_seller_customer_df, on ='order_id')
        return get_training_data_df


