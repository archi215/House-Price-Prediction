import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(page_title="Charts", page_icon="📊", layout="wide")
st.title("📊 House Price Market Analysis")

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    df['price'] = np.expm1(df['price'])
    return df

df = load_data()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Houses",  f"{len(df):,}")
col2.metric("Average Price", f"${df['price'].mean():,.0f}")
col3.metric("Lowest Price",  f"${df['price'].min():,.0f}")
col4.metric("Highest Price", f"${df['price'].max():,.0f}")

st.divider()
st.write("### 💰 Price Distribution")
fig1 = px.histogram(df, x='price', nbins=50,
    labels={'price': 'House Price ($)'},
    color_discrete_sequence=['#3B82F6'])
st.plotly_chart(fig1, use_container_width=True)

st.divider()
st.write("### 📐 Price vs Living Area")
fig2 = px.scatter(df, x='sqft_living', y='price',
    labels={'sqft_living': 'Living Area (sqft)', 'price': 'Price ($)'},
    opacity=0.4, color_discrete_sequence=['#8B5CF6'])
st.plotly_chart(fig2, use_container_width=True)

st.divider()
st.write("### 🛏 Average Price by Bedrooms")
bedroom_price = df[df['bedrooms'] <= 8].groupby('bedrooms')['price'].mean().reset_index()
fig3 = px.bar(bedroom_price, x='bedrooms', y='price',
    labels={'bedrooms': 'Bedrooms', 'price': 'Average Price ($)'},
    color='price', color_continuous_scale='Blues')
st.plotly_chart(fig3, use_container_width=True)

st.divider()
st.write("### ⭐ Average Price by Condition")
condition_price = df.groupby('condition')['price'].mean().reset_index()
fig4 = px.bar(condition_price, x='condition', y='price',
    labels={'condition': 'Condition (1-5)', 'price': 'Average Price ($)'},
    color='price', color_continuous_scale='Greens')
st.plotly_chart(fig4, use_container_width=True)

st.divider()
st.write("### 🌊 Waterfront vs Non Waterfront")
waterfront_price = df.groupby('waterfront')['price'].mean().reset_index()
waterfront_price['waterfront'] = waterfront_price['waterfront'].map(
    {0: 'No Waterfront', 1: 'Waterfront'})
fig5 = px.bar(waterfront_price, x='waterfront', y='price',
    labels={'waterfront': '', 'price': 'Average Price ($)'},
    color='waterfront',
    color_discrete_sequence=['#3B82F6', '#06B6D4'])
st.plotly_chart(fig5, use_container_width=True)

st.divider()
st.caption("Data Source: Kaggle House Sales Dataset | Seattle, Washington, USA")
