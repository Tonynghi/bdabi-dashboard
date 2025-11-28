import streamlit as st
from helpers.gcs_loader import load_parquet_from_gcs

df = load_parquet_from_gcs(
    bucket_name="bdabi-group7",
    blob_name="preprocessed/preprocessed.parquet"
)

def render():
    st.dataframe(df)