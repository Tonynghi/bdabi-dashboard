import streamlit as st
import pandas as pd
import altair as alt
from helpers.gcs_loader import load_parquet_from_gcs

def render_revenue_overtime(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Revenue Over Time")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])

        completed_df = df[df['order_status'] == 'delivered']

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(selected_range, tuple):
            start_date, end_date = selected_range
        else:
            start_date = min_date
            end_date = max_date

        mask = (completed_df['purchase_date'].dt.date >= start_date) & (completed_df['purchase_date'].dt.date <= end_date)
        filtered_df = completed_df[mask]

        daily_rev = (
            filtered_df.groupby(filtered_df['purchase_date'].dt.date)['payment_value']
            .sum()
            .reset_index()
            .rename(columns={'purchase_date': 'date', 'payment_value': 'revenue'})
        )

        if daily_rev.empty:
            st.warning("No data available for selected date range.")
            return

        chart = (
            alt.Chart(daily_rev)
            .mark_line()
            .encode(
                x='date:T',
                y='revenue:Q',
                tooltip=['date:T', 'revenue:Q']
            )
            .interactive()
        )

        st.altair_chart(chart, width='stretch')

def render_product_partition(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Product Category Distribution")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        df = df[df['order_status'] == 'delivered']

        min_date = df['purchase_date'].min().date()
        max_date = df['purchase_date'].max().date()

        selected = st.date_input(
            "Select date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date
        )

        if isinstance(selected, tuple):
            start, end = selected
        else:
            start = min_date
            end = max_date

        mask = (
            (df['purchase_date'].dt.date >= start) &
            (df['purchase_date'].dt.date <= end)
        )
        df = df[mask]

        cat_rev = (
            df.groupby('product_category_name')['payment_value']
            .sum()
            .reset_index()
            .rename(columns={'payment_value': 'revenue'})
        )

        if cat_rev.empty:
            st.warning("No data available for this date range.")
            return

        pie = (
            alt.Chart(cat_rev)
            .mark_arc()
            .encode(
                theta=alt.Theta("revenue:Q", title="Revenue"),
                color=alt.Color("product_category_name:N", title="Category"),
                tooltip=[
                    alt.Tooltip("product_category_name:N", title="Category"),
                    alt.Tooltip("revenue:Q", title="Revenue", format=",")
                ],
            )
            .properties(width=400, height=400)
        )

        st.altair_chart(pie, width='stretch')

def render_product_leaderboard(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Top Product Leaderboard")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered']

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range for Leaderboards",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="leaderboard_date_range"
        )

        if isinstance(selected_range, tuple):
            start_date, end_date = selected_range
        else:
            start_date = min_date
            end_date = max_date

        mask = (completed_df['purchase_date'].dt.date >= start_date) & \
               (completed_df['purchase_date'].dt.date <= end_date)
        filtered_df = completed_df[mask]

        if filtered_df.empty:
            st.warning("No data available for the selected date range.")
            return

        revenue_leaderboard = (
            filtered_df.groupby('product_category_name')['payment_value']
            .sum()
            .reset_index()
            .rename(columns={'payment_value': 'Total Revenue'})
            .sort_values(by='Total Revenue', ascending=False)
            .head(10)
        )
        
        volume_leaderboard = (
            filtered_df.groupby('product_category_name')['order_item_id']
            .count()
            .reset_index()
            .rename(columns={'order_item_id': 'Quantity Sold'})
            .sort_values(by='Quantity Sold', ascending=False)
            .head(10)
        )

        col_rev, col_vol = st.columns(2)
        
        with col_rev:
            st.markdown("### Top 10 by Revenue")
            st.dataframe(
                revenue_leaderboard,
                column_config={
                    "product_category_name": "Product",
                    "Total Revenue": st.column_config.NumberColumn("Total Revenue", format="$ %.2f"),
                },
                hide_index=True,
                width='stretch'
            )

        with col_vol:
            st.markdown("### Top 10 by Quantity Sold")
            st.dataframe(
                volume_leaderboard,
                column_config={
                    "product_category_name": "Product",
                    "Quantity Sold": st.column_config.NumberColumn("Quantity Sold", format="%d units"),
                },
                hide_index=True,
                width='stretch'
            )