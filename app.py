# app.py

import streamlit as st
import pandas as pd
from recommend import recommend, generate_rules

st.set_page_config(page_title="Market Basket System", layout="centered")

st.title("Market Basket Recommendation System")

# Load product list
df = pd.read_csv("cleaned_retail.csv")
products = df['Description'].value_counts().head(20).index.tolist()

selected_product = st.selectbox("Select a Product:", products)


if st.button("Get Recommendations"):

    try:
        results = recommend(selected_product)

        if results.empty:
            st.warning("No strong recommendations found.")
        else:
            st.success("Top Recommended Products:")
            st.dataframe(results)

    except FileNotFoundError:
        st.error("Please generate rules first.")
