import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(
    page_title="Investment Calculator",
    page_icon="💹",
    layout="centered"
)

st.title("💹 Investment Calculator")
st.write("See how much profit you can make from a house investment")
st.divider()

st.write("### 🏠 House Details")
purchase_price = st.number_input(
    "Purchase Price ($)", 100000, 5000000, 450000, step=10000
)

col1, col2 = st.columns(2)
with col1:
    appreciation = st.slider(
        "Annual Appreciation Rate (%)", 1.0, 15.0, 5.0, step=0.5
    )
    years = st.slider("Investment Period (years)", 1, 30, 10)

with col2:
    down_pct     = st.slider("Down Payment (%)", 5, 50, 20)
    interest     = st.slider("Mortgage Rate (%)", 3.0, 12.0, 6.5, step=0.1)
    monthly_rent = st.number_input(
        "Monthly Rental Income ($)", 0, 20000, 2000, step=100
    )

st.divider()

if st.button("📊 Calculate Investment", use_container_width=True):

    down_payment    = purchase_price * (down_pct / 100)
    loan_amount     = purchase_price - down_payment
    monthly_rate    = (interest / 100) / 12
    months          = years * 12
    monthly_payment = (loan_amount * monthly_rate *
                      (1 + monthly_rate)**360) / \
                      ((1 + monthly_rate)**360 - 1)

    future_value   = purchase_price * ((1 + appreciation/100) ** years)
    total_profit   = future_value - purchase_price
    total_rent     = monthly_rent * months
    total_mortgage = monthly_payment * months
    net_profit     = total_profit + total_rent - total_mortgage
    roi            = (net_profit / down_payment) * 100

    st.write("### 📊 Investment Summary")
    col3, col4 = st.columns(2)

    with col3:
        st.metric("Purchase Price",     f"${purchase_price:,.0f}")
        st.metric("Future Value",       f"${future_value:,.0f}",
                  f"+${total_profit:,.0f}")
        st.metric("Total Rental Income",f"${total_rent:,.0f}")

    with col4:
        st.metric("Down Payment",       f"${down_payment:,.0f}")
        st.metric("Monthly Payment",    f"${monthly_payment:,.0f}")
        st.metric("Total Mortgage",     f"${total_mortgage:,.0f}")

    st.divider()

    if net_profit > 0:
        st.success(f"### Net Profit: ${net_profit:,.0f}")
        st.success(f"### ROI: {roi:.1f}% over {years} years")
    else:
        st.error(f"### Net Loss: ${abs(net_profit):,.0f}")
        st.error(f"### ROI: {roi:.1f}% over {years} years")

    st.divider()

    st.write("### 📈 Year by Year Growth")
    year_list    = list(range(0, years + 1))
    values       = [purchase_price * ((1 + appreciation/100)**y)
                    for y in year_list]
    cum_rent     = [monthly_rent * 12 * y for y in year_list]
    cum_mortgage = [monthly_payment * 12 * y for y in year_list]

    chart_df = pd.DataFrame({
        'Year':                year_list,
        'House Value':         values,
        'Cumulative Rent':     cum_rent,
        'Cumulative Mortgage': cum_mortgage
    })

    fig = px.line(
        chart_df,
        x='Year',
        y=['House Value', 'Cumulative Rent', 'Cumulative Mortgage'],
        title=f"Investment Growth Over {years} Years",
        labels={'value': 'Amount ($)', 'variable': 'Category'},
        color_discrete_sequence=['#3B82F6', '#10B981', '#EF4444']
    )
    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.write("### 💡 Investment Advice")
    if roi > 100:
        st.success("Excellent investment! ROI over 100% is outstanding.")
    elif roi > 50:
        st.success("Good investment! Solid returns over the period.")
    elif roi > 0:
        st.warning("Moderate investment. Consider higher appreciation areas.")
    else:
        st.error("Poor investment. Increase rental income or reduce costs.")

st.divider()
st.caption("Investment calculations are estimates only — not financial advice")