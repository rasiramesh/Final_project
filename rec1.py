import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Function to load data with caching
@st.experimental_memo
def load_data():
    data = pd.read_csv(r'C:\Users\Hp\Desktop\final  project\Womens Fashions.csv')
    # Assuming 'name' is the primary descriptor of products
    data['name'] = data['name'].fillna('')
    return data

# Function to recommend products
def recommend_products(selected_product, data, n_recommendations=5):
    # Using TF-IDF Vectorizer on product names
    tfidf = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['name'])

    # Compute cosine similarity
    cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

    # Get the index of the product that matches the name
    idx = data.index[data['name'] == selected_product].tolist()[0]

    # Get the pairwise similarity scores of all products with that product
    sim_scores = list(enumerate(cosine_sim[idx]))

    # Sort the products based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the n most similar products
    sim_scores = sim_scores[1:n_recommendations+1]

    # Get the product indices
    product_indices = [i[0] for i in sim_scores]

    # Return the top n most similar products and their image URLs
    recommended_titles = data['name'].iloc[product_indices]
    recommended_images = data['image'].iloc[product_indices] if 'image' in data.columns else [None] * n_recommendations
    return recommended_titles, recommended_images

def main():
    st.title("Women's Fashion Product Recommendation System")

    data = load_data()

    # Dropdown to select a product
    product_list = data['name'].tolist()
    selected_product = st.selectbox("Select a Product", product_list)

    if st.button("Get Recommendation"):
        recommendations, images = recommend_products(selected_product, data)
        for title, image_url in zip(recommendations, images):
            st.write(title)
            if image_url:
                st.image(image_url)

if __name__ == "__main__":
    main()
