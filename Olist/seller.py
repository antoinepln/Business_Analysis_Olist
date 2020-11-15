import pandas as pd
import numpy as np
from olist.data import Olist
from olist.order import Order
from olist.utils import haversine_distance, dim_five_star, dim_one_star


class Seller:

    def __init__(self):
        # Import only data once
        olist = Olist()
        self.data = olist.get_data()
        self.matching_table = olist.get_matching_table()
        self.order = Order()

    def get_seller_features(self):
        """
        Returns a DataFrame with:
       'seller_id', 'seller_city', 'seller_state'
        """
        sellers = self.data['sellers']
        get_seller_features_df = sellers[['seller_id','seller_city', 'seller_state']]
        get_seller_features_df.set_index('seller_id',inplace = True)
        return get_seller_features_df

    def get_seller_delay_wait_time(self):
        """
        Returns a DataFrame with:
       'seller_id', 'delay_to_carrier', 'seller_wait_time'
        """
        items = self.data['order_items']
        orders = self.data['orders']
        orders['order_delivered_carrier_date'] = pd.to_datetime(orders['order_delivered_carrier_date'])
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        items['shipping_limit_date'] = pd.to_datetime(items['shipping_limit_date'])
        get_seller_delay = items.merge(orders, on = "order_id").query('order_status == "delivered"')
        delay_to_carrier = get_seller_delay['shipping_limit_date'] - get_seller_delay['order_delivered_carrier_date']
        delay_to_carrier = delay_to_carrier.map(lambda x : x.days)
        seller_wait_time = get_seller_delay['order_delivered_customer_date'] - get_seller_delay['order_purchase_timestamp']
        seller_wait_time = seller_wait_time.map(lambda x : x.days)
        get_seller_delay['seller_wait_time'] = seller_wait_time
        get_seller_delay['delay_to_carrier'] = delay_to_carrier
        get_seller_delay = get_seller_delay[['seller_id', 'delay_to_carrier', 'seller_wait_time']]
        get_seller_delay.set_index('seller_id',inplace = True)
        return get_seller_delay

    def get_active_dates(self):
        """
        Returns a DataFrame with:
       'seller_id', 'date_first_sale', 'date_last_sale'
        """
        items = self.data['order_items']
        orders = self.data['orders']
        get_active_date = items.merge(orders, on = "order_id").query('order_status == "delivered"')
        get_active_date = get_active_date[['seller_id','order_purchase_timestamp']]
        date_last_sale_df = get_active_date.groupby(by = 'seller_id').max()
        date_first_sale_df = get_active_date.groupby(by = 'seller_id').min()
        get_active_date = date_first_sale_df.merge(date_last_sale_df, on = 'seller_id')
        return get_active_date

    def get_review_score(self):
        """
        Returns a DataFrame with:
        'seller_id', 'share_of_five_stars', 'share_of_one_stars',
        'review_score'
        """
        items = self.data['order_items']
        reviews = self.data['order_reviews']
        get_review_score = reviews.merge(items, on = "order_id")[["seller_id","review_score"]]
        get_review_score["dim_is_five_star"] = get_review_score["review_score"].map(dim_five_star)
        get_review_score["dim_is_one_star"] = get_review_score["review_score"].map(dim_one_star)
        five_star_df = get_review_score.groupby(by = 'seller_id').sum()[['dim_is_five_star']]
        one_star_df = get_review_score.groupby(by = 'seller_id').sum()[['dim_is_one_star']]
        number_orders = self.matching_table.groupby(by = 'seller_id').nunique()[['order_id']]
        number_orders.columns = ['number_orders']
        five_star_df['dim_is_five_star'] = five_star_df['dim_is_five_star'] / number_orders['number_orders']
        one_star_df['dim_is_one_star'] = one_star_df['dim_is_one_star'] / number_orders['number_orders']
        get_review_score_df = get_review_score.groupby(by = "seller_id").mean()[["review_score"]]
        get_review_score_df['share_of_five_star'] = five_star_df['dim_is_five_star']
        get_review_score_df['share_of_one_star'] = one_star_df['dim_is_one_star']
        get_review_score_df = get_review_score_df[['share_of_five_star', 'share_of_one_star', 'review_score']]
        return get_review_score_df


    def get_quantity(self):
        """
        Returns a DataFrame with:
        'seller_id', 'n_orders', 'quantity'
        """
        items = self.data['order_items']
        orders = self.data['orders']
        get_quantity_df = self.matching_table.groupby(by = 'seller_id').nunique()[['order_id']]
        get_quantity_df.columns = ['n_orders']
        quantity = items.groupby(by = 'seller_id').count()[['product_id']]
        get_quantity_df['quantity'] = quantity
        return get_quantity_df

    def get_sales(self):
        """
        Returns a DataFrame with:
        'seller_id', 'sales'
        """
        items = self.data['order_items']
        sales = items['price'] - items['freight_value']
        items['sales'] = sales
        get_sales_df = items.groupby('seller_id').sum()[['price']]
        get_sales_df.columns = ['sales']
        return get_sales_df

    def get_training_data(self):
        """
        Returns a DataFrame with:
        'seller_id', 'seller_state', 'seller_city', 'delay_to_carrier',
        'seller_wait_time', 'share_of_five_stars', 'share_of_one_stars',
        'seller_review_score', 'n_orders', 'quantity', 'date_first_sale', 'date_last_sale', 'sales'
        """
        seller_features_df = self.get_seller_features()
        seller_delay_wait_time_df = self.get_seller_delay_wait_time()
        active_dates_df = self.get_active_dates()
        review_score_df = self.get_review_score()
        quantity_df = self.get_quantity()
        sales_df = self.get_sales()
        get_training_data_df = seller_features_df.merge(seller_delay_wait_time_df, on = 'seller_id')
        get_training_data_df = get_training_data_df.merge(active_dates_df, on='seller_id')
        get_training_data_df = get_training_data_df.merge(review_score_df, on ='seller_id')
        get_training_data_df = get_training_data_df.merge(quantity_df, on ='seller_id')
        get_training_data_df = get_training_data_df.merge(sales_df, on ='seller_id')
        return get_training_data_df


