import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="SIGAP - Smart Risk Analytics", layout="wide")

# Membaca dan merender file index.html
with open("index.html", "r", encoding="utf-8") as f:
    html_content = f.read()

components.html(html_content, height=1200, scrolling=True)