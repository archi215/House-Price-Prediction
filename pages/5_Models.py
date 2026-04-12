import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
import plotly.express as px

st.set_page_config(
    page_title="Model Comparison",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Model Comparison")
st.write("Compare XGBoost against other ML models")
st.divider()

# ── Extended results from model_comparison.py ─────────────────────────────────
st.write("### 📊 All 12 Models Compared")

extended_results = {
    'Model': [
        'Gradient Boosting', 'XGBoost', 'Random Forest',
        'Extra Trees', 'Decision Tree', 'ElasticNet',
        'Lasso', 'Ridge', 'KNN', 'Linear Regression',
        'AdaBoost', 'SVR'
    ],
    'R² Score': [
        0.75, 0.71, 0.70, 0.69, 0.64,
        0.60, 0.60, 0.59, 0.59, 0.58, 0.56, 0.33
    ],
    'MAE ($)': [
        84535, 92187, 94004, 97592, 105490,
        105938, 106675, 106200, 114733,
        106348, 117263, 134660
    ],
    'RMSE ($)': [
        139556, 149711, 152379, 155500, 166701,
        176616, 176791, 179263, 179572,
        180774, 185071, 228543
    ]
}

ext_df = pd.DataFrame(extended_results)

# ── Full comparison table ─────────────────────────────────────────────────────
st.dataframe(ext_df, use_container_width=True)

st.divider()

# ── R² Score chart ────────────────────────────────────────────────────────────
st.write("### 🏆 R² Score Comparison (Higher is Better)")
fig1 = px.bar(
    ext_df,
    x='Model',
    y='R² Score',
    color='R² Score',
    color_continuous_scale='Blues',
    title='R² Score — All 12 Models'
)
fig1.update_layout(xaxis_tickangle=-45)
fig1.add_hline(
    y=0.7,
    line_dash="dash",
    annotation_text="Good threshold (0.70)"
)
st.plotly_chart(fig1, use_container_width=True)

st.divider()

# ── MAE Chart ─────────────────────────────────────────────────────────────────
st.write("### 💰 MAE Comparison (Lower is Better)")
fig2 = px.bar(
    ext_df,
    x='Model',
    y='MAE ($)',
    color='MAE ($)',
    color_continuous_scale='Reds',
    title='Mean Absolute Error — All 12 Models'
)
fig2.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# ── RMSE Chart ────────────────────────────────────────────────────────────────
st.write("### 📉 RMSE Comparison (Lower is Better)")
fig3 = px.bar(
    ext_df,
    x='Model',
    y='RMSE ($)',
    color='RMSE ($)',
    color_continuous_scale='Oranges',
    title='Root Mean Square Error — All 12 Models'
)
fig3.update_layout(xaxis_tickangle=-45)
st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── Winner ────────────────────────────────────────────────────────────────────
st.write("### 🥇 Results Summary")

col1, col2, col3 = st.columns(3)
col1.success("🥇 Best R²\nGradient Boosting\n0.75")
col2.success("🥈 Second Best\nXGBoost\n0.71")
col3.success("🥉 Third Best\nRandom Forest\n0.70")

st.divider()

# ── Conclusion ────────────────────────────────────────────────────────────────
st.write("### 📝 Conclusion")
st.info("""
**Why Tree Based Models Won:**
- House prices have complex non linear relationships
- Tree models capture these patterns better than linear models
- Ensemble methods (combining multiple trees) reduce errors

**Why Gradient Boosting beat XGBoost:**
- Our dataset is relatively small (4,500 rows)
- On larger datasets XGBoost typically wins
- Both are boosting algorithms — very similar performance

**Why SVR performed worst:**
- SVR struggles with high dimensional data (24 features)
- Better suited for small datasets with few features

**Why Linear Models all scored similarly (~0.58-0.60):**
- House prices are NOT linearly related to features
- Adding regularization (Ridge/Lasso) didn't help much
- Non linear models are clearly needed here
""")

st.divider()

# ── Bias Variance table ───────────────────────────────────────────────────────
st.write("### ⚖️ Bias-Variance Tradeoff")

bv_data = {
    'Model': [
        'Linear Regression', 'Ridge', 'Lasso', 'ElasticNet',
        'Decision Tree', 'Random Forest', 'Gradient Boosting',
        'SVR', 'KNN', 'AdaBoost', 'Extra Trees', 'XGBoost'
    ],
    'Bias': [
        'High', 'High', 'High', 'Medium',
        'Low', 'Low', 'Low',
        'Medium', 'Low', 'Low', 'Low', 'Low'
    ],
    'Variance': [
        'Low', 'Low', 'Low', 'Low',
        'High', 'Medium', 'Medium',
        'Medium', 'High', 'Medium', 'Medium', 'Low'
    ],
    'Explanation': [
        'Too simple for house prices',
        'Better than Linear — reduces overfitting',
        'Does automatic feature selection',
        'Combines Ridge and Lasso',
        'Overfits easily without depth control',
        'Good balance — reduces overfitting via averaging',
        'Learns from errors sequentially',
        'Good for small datasets only',
        'Sensitive to irrelevant features',
        'Focuses on hard to predict examples',
        'Faster than Random Forest',
        'Best balance — regularization built in'
    ]
}

bv_df = pd.DataFrame(bv_data)
st.dataframe(bv_df, use_container_width=True)

st.divider()
st.caption("Models trained on 80% data | Tested on 20% hidden data | Seattle House Dataset")