import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Correlation Heatmap",
    page_icon="🔥",
    layout="wide"
)

st.title("🔥 Correlation Heatmap")
st.write("See how each feature is related to house price")
st.divider()

@st.cache_data
def load_data():
    df = pd.read_csv("data/cleaned_data.csv")
    df['price'] = np.expm1(df['price'])
    return df

df = load_data()

# ── Select numeric columns ────────────────────────────────────────────────────
numeric_cols = [
    'price', 'bedrooms', 'bathrooms', 'sqft_living',
    'sqft_lot', 'floors', 'waterfront', 'view',
    'condition', 'sqft_above', 'sqft_basement',
    'house_age', 'total_rooms', 'living_lot_ratio'
]

corr = df[numeric_cols].corr().round(2)

# ── Full heatmap ──────────────────────────────────────────────────────────────
st.write("### 🗺 Full Correlation Matrix")
fig1 = px.imshow(
    corr,
    text_auto=True,
    color_continuous_scale='RdBu_r',
    title="Correlation Between All Features",
    aspect='auto'
)
fig1.update_layout(height=600)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── Price correlation bar chart ───────────────────────────────────────────────
st.write("### 💰 Which Features Affect Price Most?")
price_corr = corr['price'].drop('price').sort_values()

fig2 = px.bar(
    x=price_corr.values,
    y=price_corr.index,
    orientation='h',
    title="Correlation with House Price",
    labels={'x': 'Correlation Score', 'y': 'Feature'},
    color=price_corr.values,
    color_continuous_scale='RdBu_r'
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── How to read ───────────────────────────────────────────────────────────────
st.write("### 📖 How to Read This")
col1, col2, col3 = st.columns(3)
col1.success("**+1.0 (Dark Blue)**\nStrong positive\nHigher value = Higher price")
col2.warning("**0.0 (White)**\nNo relationship\nDoesn't affect price")
col3.error("**-1.0 (Dark Red)**\nStrong negative\nHigher value = Lower price")

st.divider()

# ── Top correlations ──────────────────────────────────────────────────────────
st.write("### 🔝 Top Features Positively Correlated with Price")
top_pos = corr['price'].drop('price').sort_values(ascending=False).head(5)
for feat, val in top_pos.items():
    st.write(f"- **{feat}**: {val}")

st.divider()

st.write("### 🔻 Top Features Negatively Correlated with Price")
top_neg = corr['price'].drop('price').sort_values().head(5)
for feat, val in top_neg.items():
    st.write(f"- **{feat}**: {val}")

st.divider()
st.caption("Data Source: Kaggle House Sales Dataset | Seattle, Washington")