import streamlit as st
import pandas as pd
import altair as alt
from helpers.gcs_loader import load_parquet_from_gcs
from prophet import Prophet
from prophet.plot import plot_plotly

def get_daily_revenue(df):
    df_revenue = df[df['order_status'] == 'delivered'].copy()
    daily_revenue = (
        df_revenue.groupby(df_revenue['purchase_date'].dt.date)['payment_value']
        .sum()
        .reset_index()
    )

    daily_revenue.columns = ['ds', 'y']
    daily_revenue['ds'] = pd.to_datetime(daily_revenue['ds'])

    return daily_revenue


def render_revenue_forecasting(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    with column:
        st.subheader("Revenue Forecasting")
        
        m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            interval_width=0.90 
        )

        daily_revenue = get_daily_revenue(df)

        m.fit(daily_revenue)
        future = m.make_future_dataframe(periods=180)
        forecast = m.predict(future)

        fig = plot_plotly(m, forecast)
        fig.update_layout(
            xaxis_title="Date",
            yaxis_title="Revenue (R$)"
        )
        st.plotly_chart(fig)

def map_month_to_quarter(month):
    if month in [1, 2, 3]:
        return 'Q1 (Jan-Mar)'
    elif month in [4, 5, 6]:
        return 'Q2 (Apr-Jun)'
    elif month in [7, 8, 9]:
        return 'Q3 (Jul-Sep)'
    else:
        return 'Q4 (Oct-Dec)'

def render_seasonal_segmentation(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    with column:
        st.subheader('Seasonal Product Segmentation')

        df_seasonal = df[df['order_status'] == 'delivered'].copy()
        df_seasonal['purchase_date'] = pd.to_datetime(df_seasonal['purchase_date'])

        df_seasonal['purchase_month'] = df_seasonal['purchase_date'].dt.month
        
        df_seasonal['purchase_quarter'] = df_seasonal['purchase_month'].apply(map_month_to_quarter)

        seasonal_sales = (
            df_seasonal.groupby(['purchase_quarter', 'product_category_name'])['order_item_id']
            .count()
            .reset_index(name='Sales_Volume')
            .sort_values(by=['purchase_quarter', 'Sales_Volume'], ascending=[True, False])
        )

        quarters = seasonal_sales['purchase_quarter'].unique()
        selected_quarter = st.selectbox(
            "Select Sales Quarter to View",
            options=quarters,
            index=len(quarters) - 1, # Default to the most recent quarter (Q4)
            key="seasonal_quarter_filter"
        )

        filtered_sales = seasonal_sales[
            seasonal_sales['purchase_quarter'] == selected_quarter
        ]

        st.dataframe(filtered_sales)

    