import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from helpers.gcs_loader import load_parquet_from_gcs

def render_delivery_performance(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Delivery Performance Overview")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="delivery_perf_date"
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
            st.warning("No data available for selected date range.")
            return

        avg_delivery_time = filtered_df['delivery_time'].mean()
        avg_delay = filtered_df['delivery_delay'].mean()
        on_time_pct = (filtered_df['delivery_delay'] <= 0).mean() * 100
        late_pct = (filtered_df['delivery_delay'] > 0).mean() * 100

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Avg Delivery Time", f"{avg_delivery_time:.1f} days")
        with col2:
            st.metric("Avg Delay", f"{avg_delay:.1f} days")
        with col3:
            st.metric("On-Time Rate", f"{on_time_pct:.1f}%")
        with col4:
            st.metric("Late Rate", f"{late_pct:.1f}%")

        st.markdown("### Delivery Time Distribution")
        delivery_bins = pd.cut(
            filtered_df['delivery_time'], 
            bins=[0, 7, 14, 21, 28, 35, 100],
            labels=['0-7 days', '8-14 days', '15-21 days', '22-28 days', '29-35 days', '35+ days']
        )
        
        delivery_dist = delivery_bins.value_counts().reset_index()
        delivery_dist.columns = ['Time Range', 'Count']

        chart = (
            alt.Chart(delivery_dist)
            .mark_bar()
            .encode(
                x=alt.X('Time Range:N', title='Delivery Time Range'),
                y=alt.Y('Count:Q', title='Number of Orders'),
                color=alt.Color('Time Range:N', legend=None),
                tooltip=['Time Range:N', 'Count:Q']
            )
            .properties(height=300)
        )
        st.altair_chart(chart, width='stretch')


def render_delivery_delay_analysis(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Delivery Delay Trends")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="delivery_delay_date"
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
            st.warning("No data available for selected date range.")
            return

        filtered_df['month'] = filtered_df['purchase_date'].dt.to_period('M').astype(str)
        
        monthly_delay = filtered_df.groupby('month').agg({
            'delivery_delay': 'mean',
            'delivery_time': 'mean',
            'order_id': 'count'
        }).reset_index()
        
        monthly_delay.columns = ['Month', 'Avg Delay', 'Avg Delivery Time', 'Order Count']

        delay_chart = (
            alt.Chart(monthly_delay)
            .mark_line(point=True)
            .encode(
                x=alt.X('Month:N', title='Month'),
                y=alt.Y('Avg Delay:Q', title='Average Delay (days)'),
                tooltip=['Month:N', 'Avg Delay:Q', 'Order Count:Q']
            )
            .properties(height=300)
        )
        
        st.altair_chart(delay_chart, width='stretch')

        st.markdown("### Delay Categories")
        delay_categories = pd.cut(
            filtered_df['delivery_delay'],
            bins=[-100, 0, 7, 14, 30, 100],
            labels=['On Time', '1-7 days late', '8-14 days late', '15-30 days late', '30+ days late']
        )
        
        delay_dist = delay_categories.value_counts().reset_index()
        delay_dist.columns = ['Category', 'Count']
        delay_dist['Percentage'] = (delay_dist['Count'] / delay_dist['Count'].sum() * 100).round(2)

        st.dataframe(
            delay_dist,
            column_config={
                "Category": "Delay Category",
                "Count": st.column_config.NumberColumn("Orders", format="%d"),
                "Percentage": st.column_config.NumberColumn("Percentage", format="%.2f%%"),
            },
            hide_index=True,
            width='stretch'
        )


def render_delivery_by_state(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Delivery Performance by State")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="delivery_state_date"
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
            st.warning("No data available for selected date range.")
            return

        state_delivery = filtered_df.groupby('customer_state').agg({
            'delivery_time': 'mean',
            'delivery_delay': 'mean',
            'order_id': 'count'
        }).reset_index()
        
        state_delivery.columns = ['State', 'Avg Delivery Time', 'Avg Delay', 'Orders']
        state_delivery = state_delivery.sort_values('Orders', ascending=False).head(15)

        chart = (
            alt.Chart(state_delivery)
            .mark_bar()
            .encode(
                x=alt.X('State:N', title='Customer State', sort='-y'),
                y=alt.Y('Avg Delivery Time:Q', title='Avg Delivery Time (days)'),
                color=alt.Color('Avg Delay:Q', 
                               scale=alt.Scale(scheme='redyellowgreen', reverse=True),
                               title='Avg Delay (days)'),
                tooltip=['State:N', 'Avg Delivery Time:Q', 'Avg Delay:Q', 'Orders:Q']
            )
            .properties(height=400)
        )
        
        st.altair_chart(chart, width='stretch')

        st.markdown("### Top 15 States by Order Volume")
        st.dataframe(
            state_delivery,
            column_config={
                "State": "State",
                "Avg Delivery Time": st.column_config.NumberColumn("Avg Delivery Time", format="%.1f days"),
                "Avg Delay": st.column_config.NumberColumn("Avg Delay", format="%.1f days"),
                "Orders": st.column_config.NumberColumn("Total Orders", format="%d"),
            },
            hide_index=True,
            width='stretch'
        )


def render_freight_analysis(column):
    # freight cost = shipping cost
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Freight Cost Analysis")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="freight_date"
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
            st.warning("No data available for selected date range.")
            return

        avg_freight = filtered_df['freight_value'].mean()
        total_freight = filtered_df['freight_value'].sum()
        freight_ratio = (filtered_df['freight_value'] / filtered_df['price']).mean() * 100

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Avg Freight Cost", f"${avg_freight:.2f}")
        with col2:
            st.metric("Total Freight Revenue", f"${total_freight:,.2f}")
        with col3:
            st.metric("Avg Freight/Price Ratio", f"{freight_ratio:.1f}%")

        st.markdown("### Freight Cost by Product Category")
        
        category_freight = filtered_df.groupby('product_category_name').agg({
            'freight_value': ['mean', 'sum'],
            'order_id': 'count'
        }).reset_index()
        
        category_freight.columns = ['Category', 'Avg Freight', 'Total Freight', 'Orders']
        category_freight = category_freight.sort_values('Total Freight', ascending=False).head(15)

        chart = (
            alt.Chart(category_freight)
            .mark_bar()
            .encode(
                x=alt.X('Category:N', title='Product Category', sort='-y'),
                y=alt.Y('Avg Freight:Q', title='Average Freight Cost ($)'),
                color=alt.Color('Orders:Q', scale=alt.Scale(scheme='blues')),
                tooltip=['Category:N', 'Avg Freight:Q', 'Total Freight:Q', 'Orders:Q']
            )
            .properties(height=400)
        )
        
        st.altair_chart(chart, width='stretch')
