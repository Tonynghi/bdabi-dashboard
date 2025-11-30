import streamlit as st
import pandas as pd
import altair as alt
from helpers.gcs_loader import load_parquet_from_gcs
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go

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

def get_forecast(df):
    m = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=True,
            interval_width=0.90 
        )

    daily_revenue = get_daily_revenue(df)

    m.fit(daily_revenue)
    future = m.make_future_dataframe(periods=180)
    forecast = m.predict(future)

    return forecast, m

def render_revenue_forecasting(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    with column:
        st.subheader("Revenue Forecasting")

        forecast, m = get_forecast(df)

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
            index=len(quarters) - 1,
            key="seasonal_quarter_filter"
        )

        filtered_sales = seasonal_sales[
            seasonal_sales['purchase_quarter'] == selected_quarter
        ]

        st.dataframe(filtered_sales)

def render_key_forecast_metris(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])
    
    with column:
        daily_revenue = get_daily_revenue(df)
        forecast, m = get_forecast(df)

        st.subheader("Key Forecast Metrics")
    
        last_actual_date = daily_revenue['ds'].max()
        future_forecast = forecast[forecast['ds'] > last_actual_date]
        total_forecasted_revenue = future_forecast['yhat'].sum()
        
        final_actual_value = daily_revenue['y'].iloc[-1]
        final_forecasted_value = future_forecast['yhat'].iloc[-1]
        predicted_growth = ((final_forecasted_value - final_actual_value) / final_actual_value) * 100
        
        future_forecast['forecast_month'] = future_forecast['ds'].dt.to_period('M')
        monthly_yhat = future_forecast.groupby('forecast_month')['yhat'].sum()
        peak_month = monthly_yhat.idxmax().strftime('%b %Y')
        peak_revenue = monthly_yhat.max()
        
        col_total_forecast_revenue = st.container()
        col_total_forecast_revenue.metric(
            label=f"Total Forecast Revenue ({len(future_forecast)} days)",
            value=f"R$ {total_forecasted_revenue:,.0f}"
        )

        col_predicted_growth_rate = st.container()
        col_predicted_growth_rate.metric(
            label="Predicted Growth Rate",
            value=f"{predicted_growth:.2f}%",
            delta="Change from last actual value"
        )

        col_predicted_peak_month = st.container()
        col_predicted_peak_month.metric(
            label=f"Peak Predicted Month",
            value=peak_month,
            delta=f"R$ {peak_revenue:,.0f}"
        )
        