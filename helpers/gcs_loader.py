import pandas as pd
import streamlit as st
import tempfile
import os

from google.cloud import storage

@st.cache_data
def load_parquet_from_gcs(bucket_name: str, blob_name: str):
    client = storage.Client.from_service_account_info(
        st.secrets["gcp_service_account"]
    )

    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)

    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_path = tmp.name
    tmp.close()

    blob.download_to_filename(tmp_path)

    df = pd.read_parquet(tmp_path)

    os.remove(tmp_path)

    return df