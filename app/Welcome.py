import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.sidebar.success("Select a page above.")

# Specify utf-8 encoding to avoid UnicodeDecodeError
with open("README.md", "r", encoding="utf-8") as f:
    st.markdown(f.read())
