import streamlit as st

from features import sales_performance;

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

def render_dashboard():
    tab_sales_performance, tab_sales_forecasting, tab_customer_behaviours, tab_geographic_insights, tab_delivery, tab_customer_churn_prediction, tab_fraud_detection = st.tabs(tab_names)

    with tab_sales_performance:
        st.header("Sales Performance")
        col_revenue_overtime, col_product_distribution = st.columns(2)
        sales_performance.render_revenue_overtime(col_revenue_overtime)
        sales_performance.render_product_partition(col_product_distribution)
        
        col_product_leaderboard = st.container()
        sales_performance.render_product_leaderboard(col_product_leaderboard)



render_dashboard()