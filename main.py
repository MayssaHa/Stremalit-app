import streamlit as st
from streamlit_multipage import MultiPage


from pages import pages

app = MultiPage()
app.st = st

for app_name, app_function in pages.items():
    app.add_app(app_name, app_function)

app.run()