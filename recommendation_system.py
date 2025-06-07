# ai_powered_product_recommendation_system.py
# Amazon Product Recommendation System

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer
import os
from scipy.sparse import coo_matrix

# Load dataset
train_data = pd.read_csv('/kaggle/input/amazon-sales-dataset/amazon.csv')
train_data.columns

# Download dataset using kagglehub
import kagglehub
karkavelrajaj_amazon_sales_dataset_path = kagglehub.dataset_download('karkavelrajaj/amazon-sales-dataset')
print('Data source import complete.')

# Preview data
train_data.head(2)

# Select relevant columns
train_data = train_data[['user_id','product_id', 'rating', 'rating_count', 'category', 'product_name', 'img_link', 'about_product']]
train_data.head(3)
train_data['about_product']
train_data.shape
train_data.isnull().sum()

# Fill missing rating_count values
train_data.loc[:, 'rating_count'] = train_data['rating_count'].fillna(0)
train_data.isnull().sum()

# Remove duplicates
train_data.duplicated().sum()
train_data_cleaned = train_data.drop_duplicates(subset=['user_id', 'product_id'])
train_data_cleaned

# Rename columns for clarity
column_name_mapping = {
    'user_id': 'ID',
    'product_id': 'ProdID',
    'rating': 'Rating',
    'rating_count': 'ReviewCount',
    'category': 'Category',
    'product_name': 'Name',
    'img_link': 'ImageURL',
    'about_product': 'Description'
}
train_data.rename(columns=column_name_mapping, inplace=True)
train_data

# Encode user and product IDs
from sklearn.preprocessing import LabelEncoder
le_id = LabelEncoder()
le_prod = LabelEncoder()
train_data['ID']= le_id.fit_transform(train_data['ID'])
train_data['ProdID'] = le_prod.fit_transform(train_data['ProdID'])
train_data

# Check for duplicates again
train_data.duplicated().sum()

# Basic statistics
num_users = train_data['ID'].nunique()
num_items = train_data['ProdID'].nunique()
num_ratings = train_data['Rating'].nunique()
print(f"Number of unique users: {num_users}")
print(f"Number of unique items: {num_items}")
print(f"Number of unique ratings: {num_ratings}")

# Plot distributions
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
train_data['ID'].value_counts().hist(bins=10, edgecolor='k')
plt.xlabel('Interactions per User')
plt.ylabel('Number of Users')
plt.title('Distribution of Interactions per User')
plt.subplot(1, 2, 2)
train_data['ProdID'].value_counts().hist(bins=10, edgecolor='k',color='green')
plt.xlabel('Interactions per Item')
plt.ylabel('Number of Items')
plt.title('Distribution of Interactions per Item')
plt.tight_layout()
plt.show()

# Plot rating distribution
train_data['Rating'].value_counts().plot(kind='bar',color='Black')

# --- Text Preprocessing for Content-Based Filtering ---
import spacy
from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm")

def clean_and_extract_tags(text):
    doc = nlp(text.lower())
    tags = [token.text for token in doc if token.text.isalnum() and token.text not in STOP_WORDS]
    return ', '.join(tags)

columns_to_extract_tags_from = ['Description']
for column in columns_to_extract_tags_from:
    train_data[column] = train_data[column].apply(clean_and_extract_tags)

# Clean up category column
train_data['Category'] = train_data['Category'].str.replace('|', ',')
train_data

# Combine tags for content-based filtering
columns_to_extract_tags = ['Description','Category']
train_data['Tags'] = train_data[columns_to_extract_tags].apply(lambda row: ', '.join(row), axis=1)

# Ensure ratings are numeric
train_data['Rating'] = pd.to_numeric(train_data['Rating'], errors='coerce')

# Save cleaned data
train_data.to_csv('cleaned_data.csv', index=False)
from IPython.display import FileLink
FileLink('cleaned_data.csv')

# --- Top Rated Items ---
average_ratings =train_data.groupby(['Name','ReviewCount','ImageURL'])['Rating'].mean().reset_index()
average_ratings.sort_values(by='Rating', ascending=False)
top_rated_items = average_ratings.sort_values(by='Rating', ascending=False)
rating_base_recommendation = top_rated_items.head(15)
rating_base_recommendation

# --- Content-Based Recommendation ---
train_data.head(2)
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
tfidf_vectorizer = TfidfVectorizer(stop_words='english')
tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
cosine_similarities_content = cosine_similarity(tfidf_matrix_content,tfidf_matrix_content)
cosine_similarities_content

# Example: Get similar items for a given product
train_data['Name'][0]
item_name = 'Wayona Nylon Braided USB to Lightning Fast Charging and Data Sync Cable Compatible for iPhone 13, 12,11, X, 8, 7, 6, 5, iPad Air, Pro, Mini (3 FT Pack of 1, Grey)'
item_index = train_data[train_data['Name']==item_name].index[0]
similar_items = list(enumerate(cosine_similarities_content[item_index]))
sorted(similar_items, key=lambda x:x[1], reverse=True)
similar_items = sorted(similar_items, key=lambda x:x[1], reverse=True)
top_similar_items = similar_items[1:10]
recommended_items_indics = [x[0] for x in top_similar_items]
train_data.iloc[recommended_items_indics][['Name','ReviewCount','Rating']]

# Content-based recommendation function
def content_based_recommendations(train_data, item_name, top_n=10):
    if item_name not in train_data['Name'].values:
        print(f"Item '{item_name}' not found in the training data.")
        return pd.DataFrame()
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix_content = tfidf_vectorizer.fit_transform(train_data['Tags'])
    cosine_similarities_content = cosine_similarity(tfidf_matrix_content, tfidf_matrix_content)
    item_index = train_data[train_data['Name'] == item_name].index[0]
    similar_items = list(enumerate(cosine_similarities_content[item_index]))
    similar_items = sorted(similar_items, key=lambda x: x[1], reverse=True)
    top_similar_items = similar_items[1:top_n+1]
    recommended_item_indices = [x[0] for x in top_similar_items]
    recommended_items_details = train_data.iloc[recommended_item_indices][['Name', 'ReviewCount', 'ImageURL', 'Rating']]
    return recommended_items_details

# Example usage of content-based recommendations
item_name = 'AmazonBasics Flexible Premium HDMI Cable (Black, 4K@60Hz, 18Gbps), 3-Foot'
content_based_rec = content_based_recommendations(train_data, item_name, top_n=10)
content_based_rec
item_name = 'LG 80 cm (32 inches) HD Ready Smart LED TV 32LM563BPTC (Dark Iron Gray)'
content_based_rec = content_based_recommendations(train_data, item_name, top_n=8)
content_based_rec

# --- Collaborative Filtering ---
filtered_data = train_data[train_data['ProdID'] == 346]
train_data
user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
user_item_matrix
user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating',aggfunc='mean').fillna(0).astype(int)
user_item_matrix
user_similarity = cosine_similarity(user_item_matrix)
user_similarity

target_user_id = 2
target_user_index = user_item_matrix.index.get_loc(target_user_id)
user_similarities = user_similarity[target_user_index]
user_similarities
similar_user_indices = user_similarities.argsort()[::-1][1:]
recommend_items = []
for user_index in similar_user_indices:
    rated_by_similar_user = user_item_matrix.iloc[user_index]
    not_rated_by_target_user = (rated_by_similar_user==0) & (user_item_matrix.iloc[target_user_index]==0)
    recommend_items.extend(user_item_matrix.columns[not_rated_by_target_user][:10])
recommended_items_details = train_data[train_data['ProdID'].isin(recommend_items)][['Name','ReviewCount','ImageURL','Rating']]
recommended_items_details.head(10)

def collaborative_filtering_recommendations(train_data, target_user_id, top_n=10):
    user_item_matrix = train_data.pivot_table(index='ID', columns='ProdID', values='Rating', aggfunc='mean').fillna(0)
    user_similarity = cosine_similarity(user_item_matrix)
    target_user_index = user_item_matrix.index.get_loc(target_user_id)
    user_similarities = user_similarity[target_user_index]
    similar_users_indices = user_similarities.argsort()[::-1][1:]
    recommended_items = []
    for user_index in similar_users_indices:
        rated_by_similar_user = user_item_matrix.iloc[user_index]
        not_rated_by_target_user = (rated_by_similar_user == 0) & (user_item_matrix.iloc[target_user_index] == 0)
        recommended_items.extend(user_item_matrix.columns[not_rated_by_target_user][:top_n])
    return recommended_items_details.head(10)

target_user_id = 2
top_n = 5
collaborative_filtering_rec = collaborative_filtering_recommendations(train_data, target_user_id)
print(f"Top {top_n} recommendations for User {target_user_id}:")
collaborative_filtering_rec

# --- Hybrid Recommendation ---
def hybrid_recommendations(train_data,target_user_id, item_name, top_n=10):
    content_based_rec = content_based_recommendations(train_data,item_name, top_n)
    collaborative_filtering_rec = collaborative_filtering_recommendations(train_data,target_user_id, top_n)
    hybrid_rec = pd.concat([content_based_rec, collaborative_filtering_rec]).drop_duplicates()
    return hybrid_rec.head(10)

target_user_id = 12 
item_name = "Philips Daily Collection HD2582/00 830-Watt 2-Slice Pop-up Toaster (White)"
hybrid_rec = hybrid_recommendations(train_data,target_user_id, item_name, top_n=10)
print(f"Top 10 Hybrid Recommendations for User {target_user_id} and Item '{item_name}':")
hybrid_rec