import pandas as pd
import numpy as np

from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules

#load datasets as dataframes
orders = pd.read_csv('olist_orders_dataset.csv')
customers = pd.read_csv('olist_customers_dataset.csv')
geolocation = pd.read_csv('olist_geolocation_dataset.csv')
sellers = pd.read_csv('olist_sellers_dataset.csv')
payments = pd.read_csv('olist_order_payments_dataset.csv')
reviews = pd.read_csv('olist_order_reviews_dataset.csv')
product_category = pd.read_csv('product_category_name_translation.csv')

#create order_items - a new dataframe with comlumns by other dataframes
order_items = orders.merge(orders[['order_id', 'customer_id']], on='order_id', how='left')
order_items = customers.merge(customers[['customer_id', 'customer_city']], on='customer_id', how='left')
order_items = geolocation.head(500000).merge(geolocation.head(500000)[['geolocation_zip_code_prefix', 'geolocation_city']], on='geolocation_zip_code_prefix', how='left')
order_items = sellers.merge(sellers[['seller_id', 'seller_city']], on='seller_id', how='left')
order_items = payments.merge(payments[['order_id', 'payment_type']], on='order_id', how='left')
order_items = reviews.merge(reviews[['review_id', 'review_comment_message']], on='review_id', how='left')
order_items = product_category.merge(product_category[['product_category_name', 'product_category_name_english']], on='product_category_name', how='left')
order_items.drop(['product_category_name', 'product_category_name_english_x', 'product_category_name_english_y'], axis=1, inplace=True)

#add columns in order_items dataframe
order_items['customer_id'] = orders['customer_id']
order_items['customer_city'] = customers['customer_city']
order_items['geolocation_city'] = geolocation['geolocation_city']
order_items['seller_city'] = sellers['seller_city']
order_items['payment_type'] = payments['payment_type']
order_items['review_comment_message'] = reviews['review_comment_message']
order_items['product_category_name_english'] = product_category['product_category_name_english']

#transforming the categorical variables in binary vectors ("one-hot") and generating frequent intemsets by min_support
order_items_encoded = pd.get_dummies(order_items)
frequent_itemsets = apriori(order_items_encoded, min_support=0.1, use_colnames=True)

#generating association rules by confidence or lift
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.1)
rules.sort_values(by='lift', ascending=False).drop(['antecedent support', 'consequent support',	'leverage', 'conviction'], axis=1)

#create a dataframe by rules(frozenset) with antecedents, consequents and lift

know_set = {'antec': [], 'conseq': [], 'lift': [], 'class': []}

ar = rules['antecedents']
for c in range(len(rules)):
  nar = next(iter(ar[c]))
  know_set['antec'].append(nar)

cr = rules['consequents']
for c in range(len(rules)):
  ncr = next(iter(cr[c]))
  know_set['conseq'].append(ncr)

lr = rules['lift']
for c in range(len(rules)):
  know_set['lift'].append(lr[c])

for c in range(len(rules)):
  know_set['class'].append(None)
know_df = pd.DataFrame(know_set)

know_df['class'] = know_df['lift'].apply(lambda x: 'class1' if x >= 1 else 'class2')
know_df
