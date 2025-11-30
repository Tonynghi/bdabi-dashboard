# features/fraud.py
import os
import streamlit as st
import pandas as pd
import tempfile
from google.cloud import storage

MODEL_BUCKET = "bdabi-group7"
FRAUD_BLOB = "models/fraud_candidates.parquet"
RAW_BLOB = "preprocessed/preprocessed.parquet"

def generate_fraud_data(df: pd.DataFrame):
    df['order_value'] = df['payment_value']
    df['order_purchase_timestamp'] = pd.to_datetime(df['order_purchase_timestamp'])
    df = df.sort_values(['customer_unique_id', 'order_purchase_timestamp'])
    df['order_sequence'] = df.groupby('customer_unique_id').cumcount() + 1

    first = df[df['order_sequence'] == 1].copy()
    repeat = df[df['order_sequence'] >= 2].copy()

    comp = repeat.merge(
        first[['customer_unique_id', 'order_value', 'customer_zip_code_prefix']],
        on='customer_unique_id',
        suffixes=('_current', '_first')
    )

    comp['value_ratio'] = comp['order_value_current'] / (comp['order_value_first'] + 1)
    comp['different_zip'] = (
        comp['customer_zip_code_prefix_current'] != comp['customer_zip_code_prefix_first']
    )

    comp['fraud_score'] = 0
    comp.loc[(comp['value_ratio'] >= 5) & comp['different_zip'], 'fraud_score'] = 100
    comp.loc[comp['value_ratio'] >= 5, 'fraud_score'] = comp['fraud_score'].combine(75, max)
    comp.loc[(3 <= comp['value_ratio']) & (comp['value_ratio'] < 5) & comp['different_zip'], 'fraud_score'] = 50

    fraud = comp[comp['fraud_score'] >= 75].copy()
    fraud['risk_level'] = pd.cut(
        fraud['fraud_score'],
        bins=[0, 80, 99, 100],
        labels=['High Risk', 'Very High Risk', 'Critical Risk'],
        include_lowest=True
    )
    return fraud

@st.cache_resource(ttl=3600)
def load_fraud_data():
    client = storage.Client.from_service_account_info(
        st.secrets["gcp_service_account"]
    )
    bucket = client.bucket(MODEL_BUCKET)
    blob = bucket.blob(FRAUD_BLOB)

    if blob.exists():
  
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        blob.download_to_filename(tmp.name)
        df = pd.read_parquet(tmp.name)
        tmp.close()
        os.unlink(tmp.name)
        return df
    else:
    
        st.info("Fraud dataset not found → generating from raw data...")
        tmp_raw = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        bucket.blob(RAW_BLOB).download_to_filename(tmp_raw.name)
        df_raw = pd.read_parquet(tmp_raw.name)
        tmp_raw.close()
        os.unlink(tmp_raw.name)

        df_fraud = generate_fraud_data(df_raw)

        tmp_save = tempfile.NamedTemporaryFile(delete=False, suffix=".parquet")
        df_fraud.to_parquet(tmp_save.name, index=False)
        bucket.blob(FRAUD_BLOB).upload_from_filename(tmp_save.name)
        tmp_save.close()
        os.unlink(tmp_save.name)

        return df_fraud

def render_fraud_detection(container):
    with container:
        st.title("Fraud & Anomaly Order Detection")
        df = load_fraud_data()

        st.metric("Total Fraud Cases", f"{len(df):,}")
        st.markdown("### Top 20 Most Suspicious Orders")
        if not df.empty:
            top20 = df.nlargest(20, "value_ratio")[[
                'order_id', 'customer_unique_id',
                'order_value_current', 'order_value_first',
                'value_ratio', 'different_zip', 'risk_level'
            ]]
            st.dataframe(top20, use_container_width=True)
            csv = df.to_csv(index=False)
            st.download_button("Download full fraud dataset", csv, "fraud_cases.csv", "text/csv")
        else:
            st.warning("No fraud cases found")
