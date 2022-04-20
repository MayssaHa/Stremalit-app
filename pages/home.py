import pandas as pd
import streamlit as st
from streamlit_pages.streamlit_pages import MultiPage

from PIL import Image

st.set_page_config(layout="wide")

imagee = Image.open("pages/My project.png")
imagee1 = Image.open('pages/1.jpg')


def home(st, **data):

    title_container = st.container()
    col1, col2, col3 = st.columns([1, 20, 1])

    with title_container:
        with col1:
            st.write("")
        with col2:
            st.image(imagee, width=1500)

        with col3:
            st.write("")

    st.markdown("<h2 style='text-align: center; color: #355070;'> P2M Project: Marketing Analysis System </h2>",
                unsafe_allow_html=True)
    st.markdown("""---""")
    st.markdown(
        "<h3 style='text-align: center; color: #355070;'> Project realized by: Haddar Mayssa - Ismail Khouloud : INDP2-A </h3>",
        unsafe_allow_html=True)
    st.markdown("""---""")

    col1, col2, col3, col4 = st.columns([1, 2.7, 2.7, 2.7])
    col1.write("")
    col2.markdown("<h3 style='text-align: left; color: #6D597A;'>Hotels And Tourism</h3>", unsafe_allow_html=True)

    col3.markdown("<h3 style='text-align: left; color: #6D597A;'>Food Retail Stores</h3>", unsafe_allow_html=True)

    col4.markdown("<h3 style='text-align: left; color: #6D597A;'>Banking Business</h3>", unsafe_allow_html=True)

    st.write("")
    st.markdown("    ")
    title_container = st.container()
    col1, col2, col3 = st.columns([1, 20, 1])

    with title_container:
        with col1:
            st.write("")
        with col2:
            st.image(imagee1, width=1500)

        with col3:
            st.write("")


