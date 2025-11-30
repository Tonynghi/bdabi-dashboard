import streamlit as st
from features import sales_performance, sales_forecasting, churn, fraud
from features import customer_behaviours;
from features import delivery;
from features import geographic_insight;

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
        col_key_forecast_metrics,col_revenue_forecasting = st.columns([1,3])
        sales_forecasting.render_revenue_forecasting(col_revenue_forecasting)
        sales_forecasting.render_key_forecast_metris(col_key_forecast_metrics)

elif selected_tab == "Customer Behaviours":
        st.header("Customer Behaviours")
        col_customer_loyalty, col_sales_volumes_by_reviews  = st.columns(2)
        customer_behaviours.render_customer_loyalty(col_customer_loyalty)
        customer_behaviours.render_sales_volumes_by_reviews(col_sales_volumes_by_reviews)

        col_payment_analysis = st.container()
        customer_behaviours.render_payment_analysis(col_payment_analysis)

elif selected_tab == "Geographic Insights":
        st.header("Geographic Insights")
        col_sales_region, col_customer_dist = st.columns(2)
        geographic_insight.render_sales_by_region(col_sales_region)
        geographic_insight.render_customer_distribution(col_customer_dist)

        col_seller_perf = st.container()
        geographic_insight.render_seller_performance_by_region(col_seller_perf)

        col_city_analysis = st.container()
        geographic_insight.render_city_level_analysis(col_city_analysis)

        col_product_pref = st.container()
        geographic_insight.render_regional_product_preferences(col_product_pref)

elif selected_tab == "Delivery":
        st.header("Delivery Performance")
        col_delivery_perf, col_delay_analysis = st.columns(2)
        delivery.render_delivery_performance(col_delivery_perf)
        delivery.render_delivery_delay_analysis(col_delay_analysis)

        col_delivery_state = st.container()
        delivery.render_delivery_by_state(col_delivery_state)

        col_freight = st.container()
        delivery.render_freight_analysis(col_freight)

elif selected_tab == "Customer Churn Prediction":
    churn.render_churn_prediction(st.container())

elif selected_tab == "Fraud Detection":
    fraud.render_fraud_detection(st.container())



