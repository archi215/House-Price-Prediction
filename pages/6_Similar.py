import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(
    page_title="Similar Houses",
    page_icon="🏘",
    layout="wide"
)

st.title("🏘 Similar Houses Finder")
st.write("Find houses similar to what you searched")
st.divider()

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    df['price'] = np.expm1(df['price'])
    return df

df = load_data()

st.write("### 🔍 Search Filters")
col1, col2, col3 = st.columns(3)

with col1:
    bedrooms  = st.selectbox("Bedrooms", [1,2,3,4,5], index=2)

with col2:
    condition = st.selectbox("Condition", [1,2,3,4,5], index=3)

with col3:
    max_price = st.number_input(
        "Max Price ($)", 100000, 5000000, 600000, step=50000
    )

sqft_range = st.slider(
    "Living Area Range (sqft)",
    500, 8000, (1000, 2500)
)

if st.button("🔍 Find Similar Houses", use_container_width=True):
    filtered = df[
        (df['bedrooms']    == bedrooms) &
        (df['condition']   == condition) &
        (df['price']       <= max_price) &
        (df['sqft_living'] >= sqft_range[0]) &
        (df['sqft_living'] <= sqft_range[1])
    ].copy()

    if len(filtered) == 0:
        st.warning("No houses found! Try adjusting your filters.")
    else:
        filtered = filtered.head(10)
        st.success(f"Found {len(filtered)} similar houses!")
        st.divider()

        col4, col5, col6 = st.columns(3)
        col4.metric("Average Price", f"${filtered['price'].mean():,.0f}")
        col5.metric("Lowest Price",  f"${filtered['price'].min():,.0f}")
        col6.metric("Highest Price", f"${filtered['price'].max():,.0f}")

        st.divider()

        st.write("### 🏠 Matching Houses")
        display_cols = [
            'price', 'bedrooms', 'bathrooms',
            'sqft_living', 'condition', 'house_age'
        ]
        display = filtered[display_cols].copy()
        display.columns = [
            'Price ($)', 'Beds', 'Baths',
            'Sqft', 'Condition', 'Age (yrs)'
        ]
        display['Price ($)'] = display['Price ($)'].apply(
            lambda x: f"${x:,.0f}"
        )
        st.dataframe(display, use_container_width=True)

st.divider()
st.caption("Data Source: Kaggle House Sales Dataset | Seattle, Washington")