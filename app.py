import streamlit as st
from features import sales_performance, sales_forecasting, churn, fraud

st.set_page_config(
    page_title="Business Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

tab_names = [
    "Sales Performance", 
    "Sales Forecasting", 
    "Customer Behaviours",
    "Geographic Insights",
    "Delivery",
    "Customer Churn Prediction",
    "Fraud Detection",
]


selected_tab = st.sidebar.radio("Select dashboard tab:", tab_names)

if selected_tab == "Sales Performance":
    st.header("Sales Performance")
    col1, col2 = st.columns(2)
    sales_performance.render_revenue_overtime(col1)
    sales_performance.render_product_partition(col2)
    sales_performance.render_product_leaderboard(st.container())

elif selected_tab == "Sales Forecasting":
    st.header("Sales Forecasting")
    sales_forecasting.render_revenue_forecasting(st.container())
    sales_forecasting.render_seasonal_segmentation(st.container())

elif selected_tab == "Customer Churn Prediction":
    churn.render_churn_prediction(st.container())

elif selected_tab == "Fraud Detection":
    fraud.render_fraud_detection(st.container())
