import streamlit as st

from features import sales_performance;
from features import sales_forecasting;
from features import customer_behaviours;

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

    with tab_sales_forecasting:
        st.header("Sales Forecasting")
        col_key_forecast_metrics,col_revenue_forecasting = st.columns([1,3])
        sales_forecasting.render_revenue_forecasting(col_revenue_forecasting)
        sales_forecasting.render_key_forecast_metris(col_key_forecast_metrics)

    with tab_customer_behaviours:
        st.header("Customer Behaviours")
        col_customer_loyalty, col_sales_volumes_by_reviews  = st.columns(2)
        customer_behaviours.render_customer_loyalty(col_customer_loyalty)
        customer_behaviours.render_sales_volumes_by_reviews(col_sales_volumes_by_reviews)

        col_payment_analysis = st.container()
        customer_behaviours.render_payment_analysis(col_payment_analysis)

        col_seasonal_segmentation = st.container()
        sales_forecasting.render_seasonal_segmentation(col_seasonal_segmentation)


render_dashboard()