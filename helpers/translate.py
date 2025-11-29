import streamlit as st
from deep_translator import GoogleTranslator


@st.cache_data
def translate(text):
    return GoogleTranslator(source="pt", target="en").translate(text)