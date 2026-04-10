import streamlit as st
import joblib
import pandas as pd
import numpy as np

# ── Page config (MUST be first) ───────────────────────────────────────────────
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="centered"
)

# ── Load model and features ───────────────────────────────────────────────────
model    = joblib.load("models/property_model.pkl")
features = joblib.load("models/features.pkl")

# ── Initialize history ────────────────────────────────────────────────────────
if 'history' not in st.session_state:
    st.session_state['history'] = []

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🏠 House Price Predictor")
st.write("Enter the details of a house to get an estimated price.")
st.divider()

# ── Input form ────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    bedrooms     = st.slider("🛏 Bedrooms", 1, 10, 3)
    bathrooms    = st.slider("🚿 Bathrooms", 1.0, 6.0, 2.0, step=0.25)
    sqft_living  = st.number_input("📐 Living Area (sqft)", 500, 10000, 1600)
    sqft_lot     = st.number_input("🌳 Lot Size (sqft)", 500, 50000, 7000)
    floors       = st.selectbox("🏢 Floors", [1.0, 1.5, 2.0, 2.5, 3.0])

with col2:
    condition    = st.slider("⭐ Condition (1–5)", 1, 5, 4)
    view         = st.slider("👁 View Rating (0–4)", 0, 4, 0)
    waterfront   = st.selectbox("🌊 Waterfront?", [0, 1],
                                format_func=lambda x: "Yes" if x == 1 else "No")
    yr_built     = st.number_input("🔨 Year Built", 1900, 2024, 2000)
    yr_renovated = st.number_input("🔧 Year Renovated (0 = never)", 0, 2024, 0)

st.divider()

# ── Build features ────────────────────────────────────────────────────────────
sale_year    = 2014
sale_month   = 5
sale_quarter = 2
zipcode      = 98133
city_encoded = 500000

house_age        = sale_year - yr_built
was_renovated    = 1 if yr_renovated > 0 else 0
years_since_reno = (sale_year - yr_renovated) if yr_renovated > 0 else house_age
sqft_above       = sqft_living
sqft_basement    = 0
living_lot_ratio = sqft_living / (sqft_lot + 1)
price_per_sqft   = sqft_living / (sqft_lot + 1)
total_rooms      = bedrooms + bathrooms
basement_flag    = 0

sample = {
    'bedrooms': bedrooms, 'bathrooms': bathrooms,
    'sqft_living': sqft_living, 'sqft_lot': sqft_lot,
    'floors': floors, 'waterfront': waterfront, 'view': view,
    'condition': condition, 'sqft_above': sqft_above,
    'sqft_basement': sqft_basement, 'yr_built': yr_built,
    'yr_renovated': yr_renovated, 'sale_year': sale_year,
    'sale_month': sale_month, 'sale_quarter': sale_quarter,
    'zipcode': zipcode, 'city_encoded': city_encoded,
    'house_age': house_age, 'was_renovated': was_renovated,
    'years_since_reno': years_since_reno,
    'price_per_sqft_est': price_per_sqft,
    'total_rooms': total_rooms, 'basement_flag': basement_flag,
    'living_lot_ratio': living_lot_ratio,
}

# ── Predict button ────────────────────────────────────────────────────────────
if st.button("🔮 Predict Price", use_container_width=True):
    df = pd.DataFrame([sample])
    df = df.reindex(columns=features, fill_value=0)

    log_price = model.predict(df)[0]
    price     = np.expm1(log_price)

    # ── Price range ───────────────────────────────────────────────────────────
    low_price  = price * 0.85
    high_price = price * 1.15

    # ── Monthly mortgage ──────────────────────────────────────────────────────
    down_payment    = price * 0.20
    loan_amount     = price - down_payment
    monthly_rate    = 0.065 / 12
    months          = 360
    monthly_payment = (loan_amount * monthly_rate *
                      (1 + monthly_rate)**months) / \
                      ((1 + monthly_rate)**months - 1)

    # ── Main price display ────────────────────────────────────────────────────
    st.success(f"### 🏷 Estimated Price: ${price:,.0f}")
    st.info(f"📊 Price Range: **${low_price:,.0f}** — **${high_price:,.0f}**")

    # ── Price confidence band ─────────────────────────────────────────────────
    st.write("##### 💰 Price Confidence Band")
    col_low, col_mid, col_high = st.columns(3)
    col_low.metric("🔽 Low Estimate",  f"${low_price:,.0f}",  "-15%")
    col_mid.metric("🎯 Best Estimate", f"${price:,.0f}",      "Most likely")
    col_high.metric("🔼 High Estimate",f"${high_price:,.0f}", "+15%")

    st.divider()

    # ── Price category ────────────────────────────────────────────────────────
    if price < 300000:
        category = "🟢 Budget Friendly"
        advice   = "Great value for money! Good for first time buyers."
    elif price < 600000:
        category = "🟡 Mid Range"
        advice   = "Solid mid-range property with good resale value."
    elif price < 1000000:
        category = "🟠 Premium"
        advice   = "Premium property — expect high quality finishes."
    else:
        category = "🔴 Luxury"
        advice   = "Luxury property in an exclusive price bracket."

    st.write(f"### {category}")
    st.write(advice)

    st.divider()

    # ── Extra metrics ─────────────────────────────────────────────────────────
    col3, col4, col5 = st.columns(3)
    col3.metric("💵 Price per sqft", f"${price/sqft_living:,.0f}")
    col4.metric("🛏 Bedrooms",       f"{bedrooms}")
    col5.metric("🏚 House Age",      f"{house_age} yrs")

    st.divider()

    # ── Monthly mortgage display ──────────────────────────────────────────────
    st.write("##### 🏦 Monthly Mortgage Estimate")
    col6, col7, col8 = st.columns(3)
    col6.metric("💰 Down Payment",    f"${down_payment:,.0f}")
    col7.metric("📅 Monthly Payment", f"${monthly_payment:,.0f}")
    col8.metric("🏦 Loan Amount",     f"${loan_amount:,.0f}")

    st.divider()

    # ── Save to history ───────────────────────────────────────────────────────
    st.session_state['history'].append({
        'Predicted Price':  round(price),
        'Low Estimate':     round(low_price),
        'High Estimate':    round(high_price),
        'Bedrooms':         bedrooms,
        'Bathrooms':        bathrooms,
        'Living Area':      sqft_living,
        'Condition':        condition,
        'Year Built':       yr_built,
        'Waterfront':       'Yes' if waterfront == 1 else 'No',
        'House Age':        house_age,
        'Price/sqft':       round(price/sqft_living),
        'Monthly Payment':  round(monthly_payment),
    })
    st.info("✅ Prediction saved! Check 📋 History page.")

st.divider()
st.caption("Built with XGBoost + Streamlit | R² Score: 0.72")