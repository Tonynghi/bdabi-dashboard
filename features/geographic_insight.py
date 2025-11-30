import streamlit as st
import pandas as pd
import altair as alt
import numpy as np
from helpers.gcs_loader import load_parquet_from_gcs

def render_sales_by_region(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Sales Performance by Region")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="geo_sales_date"
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

        state_sales = filtered_df.groupby('customer_state').agg({
            'payment_value': 'sum',
            'order_id': 'nunique',
            'customer_id': 'nunique'
        }).reset_index()
        
        state_sales.columns = ['State', 'Total Revenue', 'Total Orders', 'Unique Customers']
        state_sales['Avg Order Value'] = state_sales['Total Revenue'] / state_sales['Total Orders']
        state_sales = state_sales.sort_values('Total Revenue', ascending=False)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total States", len(state_sales))
        with col2:
            top_state = state_sales.iloc[0]
            st.metric("Top State", f"{top_state['State']}")
        with col3:
            st.metric("Top State Revenue", f"${top_state['Total Revenue']:,.2f}")

        st.markdown("### Top 15 States by Revenue")
        top_states = state_sales.head(15)

        chart = (
            alt.Chart(top_states)
            .mark_bar()
            .encode(
                x=alt.X('State:N', title='State', sort='-y'),
                y=alt.Y('Total Revenue:Q', title='Total Revenue ($)'),
                color=alt.Color('Total Orders:Q', 
                               scale=alt.Scale(scheme='blues'),
                               title='Total Orders'),
                tooltip=['State:N', 'Total Revenue:Q', 'Total Orders:Q', 'Unique Customers:Q', 'Avg Order Value:Q']
            )
            .properties(height=400)
        )
        
        st.altair_chart(chart, width='stretch')

        st.dataframe(
            state_sales,
            column_config={
                "State": "State",
                "Total Revenue": st.column_config.NumberColumn("Total Revenue", format="$%.2f"),
                "Total Orders": st.column_config.NumberColumn("Total Orders", format="%d"),
                "Unique Customers": st.column_config.NumberColumn("Unique Customers", format="%d"),
                "Avg Order Value": st.column_config.NumberColumn("Avg Order Value", format="$%.2f"),
            },
            hide_index=True,
            width='stretch'
        )


def render_customer_distribution(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Customer Distribution by State")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="geo_customer_date"
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

        customer_dist = filtered_df.groupby('customer_state').agg({
            'customer_id': 'nunique',
            'order_id': 'nunique',
            'payment_value': 'sum'
        }).reset_index()
        
        customer_dist.columns = ['State', 'Unique Customers', 'Total Orders', 'Total Revenue']
        customer_dist['Orders per Customer'] = customer_dist['Total Orders'] / customer_dist['Unique Customers']
        customer_dist['Revenue per Customer'] = customer_dist['Total Revenue'] / customer_dist['Unique Customers']
        customer_dist = customer_dist.sort_values('Unique Customers', ascending=False)

        st.markdown("### Top 15 States by Customer Count")
        top_customer_states = customer_dist.head(15)

        bubble_chart = (
            alt.Chart(top_customer_states)
            .mark_circle()
            .encode(
                x=alt.X('Orders per Customer:Q', title='Orders per Customer'),
                y=alt.Y('Revenue per Customer:Q', title='Revenue per Customer ($)'),
                size=alt.Size('Unique Customers:Q', 
                             title='Customer Count',
                             scale=alt.Scale(range=[100, 2000])),
                color=alt.Color('State:N', legend=None),
                tooltip=['State:N', 'Unique Customers:Q', 'Orders per Customer:Q', 'Revenue per Customer:Q']
            )
            .properties(height=400)
        )
        
        st.altair_chart(bubble_chart, width='stretch')

        st.dataframe(
            customer_dist,
            column_config={
                "State": "State",
                "Unique Customers": st.column_config.NumberColumn("Customers", format="%d"),
                "Total Orders": st.column_config.NumberColumn("Orders", format="%d"),
                "Total Revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
                "Orders per Customer": st.column_config.NumberColumn("Orders/Customer", format="%.2f"),
                "Revenue per Customer": st.column_config.NumberColumn("Revenue/Customer", format="$%.2f"),
            },
            hide_index=True,
            width='stretch'
        )


def render_seller_performance_by_region(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Seller Performance by Region")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="geo_seller_date"
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

        seller_perf = filtered_df.groupby('seller_state').agg({
            'seller_id': 'nunique',
            'order_id': 'nunique',
            'payment_value': 'sum',
            'price': 'sum'
        }).reset_index()
        
        seller_perf.columns = ['State', 'Unique Sellers', 'Total Orders', 'Total Revenue', 'Product Value']
        seller_perf['Avg Orders per Seller'] = seller_perf['Total Orders'] / seller_perf['Unique Sellers']
        seller_perf['Avg Revenue per Seller'] = seller_perf['Total Revenue'] / seller_perf['Unique Sellers']
        seller_perf = seller_perf.sort_values('Total Revenue', ascending=False)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Seller States", len(seller_perf))
        with col2:
            st.metric("Total Active Sellers", seller_perf['Unique Sellers'].sum())
        with col3:
            st.metric("Avg Orders/Seller", f"{seller_perf['Avg Orders per Seller'].mean():.1f}")

        st.markdown("### Top 15 States by Seller Performance")
        top_seller_states = seller_perf.head(15)

        chart = (
            alt.Chart(top_seller_states)
            .mark_bar()
            .encode(
                x=alt.X('State:N', title='Seller State', sort='-y'),
                y=alt.Y('Total Revenue:Q', title='Total Revenue ($)'),
                color=alt.Color('Unique Sellers:Q', 
                               scale=alt.Scale(scheme='greens'),
                               title='Seller Count'),
                tooltip=['State:N', 'Total Revenue:Q', 'Unique Sellers:Q', 'Avg Revenue per Seller:Q']
            )
            .properties(height=400)
        )
        
        st.altair_chart(chart, width='stretch')

        st.dataframe(
            seller_perf,
            column_config={
                "State": "State",
                "Unique Sellers": st.column_config.NumberColumn("Sellers", format="%d"),
                "Total Orders": st.column_config.NumberColumn("Orders", format="%d"),
                "Total Revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
                "Product Value": st.column_config.NumberColumn("Product Value", format="$%.2f"),
                "Avg Orders per Seller": st.column_config.NumberColumn("Orders/Seller", format="%.1f"),
                "Avg Revenue per Seller": st.column_config.NumberColumn("Revenue/Seller", format="$%.2f"),
            },
            hide_index=True,
            width='stretch'
        )


def render_city_level_analysis(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("City-Level Analysis")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="geo_city_date"
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

        city_analysis = filtered_df.groupby(['customer_city', 'customer_state']).agg({
            'customer_id': 'nunique',
            'order_id': 'nunique',
            'payment_value': 'sum',
            'review_score': 'mean'
        }).reset_index()
        
        city_analysis.columns = ['City', 'State', 'Customers', 'Orders', 'Revenue', 'Avg Review Score']
        city_analysis['Avg Order Value'] = city_analysis['Revenue'] / city_analysis['Orders']
        city_analysis = city_analysis.sort_values('Revenue', ascending=False).head(20)

        st.markdown("### Top 20 Cities by Revenue")
        
        st.dataframe(
            city_analysis,
            column_config={
                "City": "City",
                "State": "State",
                "Customers": st.column_config.NumberColumn("Customers", format="%d"),
                "Orders": st.column_config.NumberColumn("Orders", format="%d"),
                "Revenue": st.column_config.NumberColumn("Revenue", format="$%.2f"),
                "Avg Review Score": st.column_config.NumberColumn("Avg Review", format="%.2f"),
                "Avg Order Value": st.column_config.NumberColumn("Avg Order Value", format="$%.2f"),
            },
            hide_index=True,
            width='stretch'
        )

        st.markdown("### Revenue vs Customer Satisfaction")
        scatter_chart = (
            alt.Chart(city_analysis)
            .mark_circle()
            .encode(
                x=alt.X('Revenue:Q', title='Total Revenue ($)', scale=alt.Scale(type='log')),
                y=alt.Y('Avg Review Score:Q', title='Average Review Score'),
                size=alt.Size('Orders:Q', title='Order Count'),
                color=alt.Color('State:N', title='State'),
                tooltip=['City:N', 'State:N', 'Revenue:Q', 'Orders:Q', 'Avg Review Score:Q']
            )
            .properties(height=400)
        )
        
        st.altair_chart(scatter_chart, width='stretch')


def render_regional_product_preferences(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )

    with column:
        st.subheader("Regional Product Preferences")

        df['purchase_date'] = pd.to_datetime(df['purchase_date'])
        completed_df = df[df['order_status'] == 'delivered'].copy()

        min_date = completed_df['purchase_date'].min().date()
        max_date = completed_df['purchase_date'].max().date()

        selected_range = st.date_input(
            "Select Date Range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="geo_product_date"
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

        states = sorted(filtered_df['customer_state'].unique())
        selected_state = st.selectbox("Select State to Analyze", states, key="state_selector")

        if selected_state:
            state_df = filtered_df[filtered_df['customer_state'] == selected_state]
            
            category_sales = state_df.groupby('product_category_name').agg({
                'payment_value': 'sum',
                'order_id': 'nunique'
            }).reset_index()
            
            category_sales.columns = ['Category', 'Revenue', 'Orders']
            category_sales = category_sales.sort_values('Revenue', ascending=False).head(10)

            st.markdown(f"### Top 10 Categories in {selected_state}")
            
            chart = (
                alt.Chart(category_sales)
                .mark_bar()
                .encode(
                    y=alt.Y('Category:N', title='Product Category', sort='-x'),
                    x=alt.X('Revenue:Q', title='Revenue ($)'),
                    color=alt.Color('Orders:Q', scale=alt.Scale(scheme='viridis')),
                    tooltip=['Category:N', 'Revenue:Q', 'Orders:Q']
                )
                .properties(height=400)
            )
            
            st.altair_chart(chart, width='stretch')

            st.markdown("### Comparison with National Average")
            
            national_dist = filtered_df.groupby('product_category_name')['payment_value'].sum()
            national_pct = (national_dist / national_dist.sum() * 100).to_dict()
            
            state_dist = state_df.groupby('product_category_name')['payment_value'].sum()
            state_pct = (state_dist / state_dist.sum() * 100).to_dict()
            
            comparison = []
            for cat in category_sales['Category'].head(10):
                comparison.append({
                    'Category': cat,
                    'State %': state_pct.get(cat, 0),
                    'National %': national_pct.get(cat, 0),
                    'Difference': state_pct.get(cat, 0) - national_pct.get(cat, 0)
                })
            
            comparison_df = pd.DataFrame(comparison)
            
            st.dataframe(
                comparison_df,
                column_config={
                    "Category": "Category",
                    "State %": st.column_config.NumberColumn("State %", format="%.2f%%"),
                    "National %": st.column_config.NumberColumn("National %", format="%.2f%%"),
                    "Difference": st.column_config.NumberColumn("Difference", format="%.2f%%"),
                },
                hide_index=True,
                width='stretch'
            )
