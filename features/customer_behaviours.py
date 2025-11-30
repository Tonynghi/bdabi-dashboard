import streamlit as st
from helpers.gcs_loader import load_parquet_from_gcs
import pandas as pd
import altair as alt


def render_customer_loyalty(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    with column:
        st.subheader("Customer Loyalty")

        delivered_df = df[df['order_status'] == 'delivered'].copy()

        orders_per_customer = (
            delivered_df.groupby('customer_unique_id')['order_id']
            .nunique() # Count the number of distinct orders
            .reset_index(name='Order_Count')
        )

        total_customers = orders_per_customer.shape[0]

        repeat_customers = orders_per_customer[orders_per_customer['Order_Count'] > 1].shape[0]
        single_purchase_customers = total_customers - repeat_customers
        repeat_rate = (repeat_customers / total_customers) * 100 if total_customers > 0 else 0

        st.metric(
            label="Customer Repeat Purchase Rate",
            value=f"{repeat_rate:.2f}%",
            delta=f"{repeat_customers:,} Repeat Buyers"
        )

        loyalty_data = pd.DataFrame({
            'Type': ['Repeat Buyers', 'Single Buyers'],
            'Count': [repeat_customers, single_purchase_customers]
        })

        pie_chart = (
            alt.Chart(loyalty_data)
            .mark_arc(outerRadius=120)
            .encode(
                theta=alt.Theta("Count", stack=True),
                color=alt.Color("Type", legend=alt.Legend(title="Customer Type")),
                tooltip=['Type', 'Count']
            )
            .properties(
                title='Customer Breakdown'
            )
        )
        st.altair_chart(pie_chart, width='stretch')

def render_payment_analysis(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    with column:
        st.subheader("Revenue & Volume by Payment Type")
        payment_df = df[df['order_status'] == 'delivered'].copy()

        payment_summary = (
            payment_df.groupby('payment_type')
            .agg(
                Total_Revenue=('payment_value', 'sum'),
                Order_Volume=('order_id', 'nunique') # Unique orders per payment type
            )
            .reset_index()
            .sort_values(by='Total_Revenue', ascending=False)
        )

        st.markdown("### Total Revenue by Payment Type")
        revenue_chart = (
            alt.Chart(payment_summary)
            .mark_bar()
            .encode(
                # Sort by revenue descending
                y=alt.Y('payment_type:N', sort='-x', title="Payment Type"),
                x=alt.X('Total_Revenue:Q', title="Revenue (R$)"),
                tooltip=[
                    'payment_type:N', 
                    alt.Tooltip('Total_Revenue:Q', title="Revenue", format="$,.2f")
                ],
                color=alt.Color('payment_type:N', legend=None)
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(revenue_chart, width='stretch')

        st.markdown("### Total Orders by Payment Type")
        volume_chart = (
            alt.Chart(payment_summary)
            .mark_bar()
            .encode(
                # Sort by volume descending
                y=alt.Y('payment_type:N', sort='-x', title="Payment Type"),
                x=alt.X('Order_Volume:Q', title="Number of Orders"),
                tooltip=['payment_type:N', alt.Tooltip('Order_Volume:Q', title="Orders", format=",")],
                color=alt.Color('payment_type:N', legend=None)
            )
            .properties(height=300)
            .interactive()
        )
        st.altair_chart(volume_chart, width='stretch')

def render_sales_volumes_by_reviews(column):
    df = load_parquet_from_gcs(
        bucket_name="bdabi-group7",
        blob_name="preprocessed/preprocessed.parquet"
    )
    df['purchase_date'] = pd.to_datetime(df['purchase_date'])

    with column:
        st.subheader("Order Volume Distribution by Review Score")

        review_df = df[
            (df['order_status'] == 'delivered') & 
            (df['review_score'].notna())
        ].copy()
        
        if review_df.empty:
            st.warning("No delivered orders with review scores available.")
            return

        score_volume = (
            review_df.groupby('review_score')['order_id']
            .nunique()
            .reset_index(name='Total_Orders')
        )
        
        score_volume['review_score'] = score_volume['review_score'].astype(int)

        chart = (
            alt.Chart(score_volume)
            .mark_bar()
            .encode(
                x=alt.X('review_score:O', title="Review Score (1-5)"),
                y=alt.Y('Total_Orders:Q', title="Total Orders"),
                tooltip=['review_score:O', alt.Tooltip('Total_Orders:Q', format=",")]
            )
            .properties(height=300)
            .interactive()
        )
        
        st.altair_chart(chart, width='stretch')