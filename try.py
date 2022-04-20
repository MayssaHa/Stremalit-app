import streamlit as st
from streamlit_multipage import MultiPage
from PIL import Image
import pickle
import pandas as pd
import numpy as np
from textblob import TextBlob
import pandas as pd
import altair as alt
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import nltk
from cProfile import label
from tkinter import PIESLICE
from streamlit_echarts import st_echarts
import matplotlib.pyplot as plt
import seaborn as sns
from operator import itemgetter

from datetime import datetime
import plotly.express as px
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from IPython.display import display
import os
pd.options.display.float_format = '{:,.1f}'.format

from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts


def hotel(st, **data):
    Titre_principal = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Phenicia Hotel</h2>'
    st.markdown(Titre_principal, unsafe_allow_html=True)

    st.subheader('\n')

    DATA_URL = "./hotel_clusters.csv"

    @st.cache(
        persist=True)  # ( If you have a different use case where the data does not change so very often, you can simply use this)
    def load_data():
        data = pd.read_csv(DATA_URL)
        return data

    hotels_data = load_data()

    df = hotels_data.loc[hotels_data['hotel'] == 'Phenicia_Hotel', :]
    a = len(df)
    print(a)

    # Percentages for Phenicia_Hotel
    # stars=[{'value': 5, 'name': 3130}, {'value': 4, 'name': 1036}, {'value': 1, 'name': 76}, {'value': 3, 'name': 355}, {'value': 2, 'name': 120}]
    # clusters=[{'value': 5, 'name': 372}, {'value': 8, 'name': 1451}, {'value': 3, 'name': 111}, {'value': 0, 'name': 449}, {'value': 4, 'name': 162}, {'value': 7, 'name': 91}, {'value': 6, 'name': 614}, {'value': 2, 'name': 690}, {'value': 1, 'name': 162}, {'value': 9, 'name': 615}]
    # Top favourite clusters=[[5, 98.65591397849462], [8, 99.24190213645761], [3, 90.09009009009009], [0, 83.51893095768375], [4, 77.77777777777779], [7, 82.41758241758241], [6, 75.2442996742671], [2, 91.88405797101449], [1, 82.09876543209876], [9, 73.8211382113821]]
    # affichage de a

    select = st.sidebar.selectbox('How would you like to be contacted?', ('Email', 'Home phone', 'Mobile phone'))
    # bech naamel affichage lel wetla eli bahdheha (infos)
    col1, col2, col3, col4, col5, col6, col7 = st.columns([0.3, 2, 0.3, 2, 0.3, 2, 0.3])

    with col1:
        st.empty()
    with col2:
        with st.container():
            st.info("The length of our database: **_111587_** reviews")
    with col3:
        st.empty()
    with col4:
        with st.container():
            st.info("Number of reviews for this hotel: **_4717_** reviews")
    with col5:
        st.empty()
    with col6:
        with st.container():
            st.info("Number of hotels in the same region (competition): **_128_**")
    with col7:
        st.empty()

    st.subheader('\n')
    st.subheader('\n')

    T1 = '<p style="color:#355070; text-align:left; text-shadow: 0 0 1px #000080; font-size: 30px; font-style:bold">Ratings & Clustering Percentages</p>'
    st.markdown(T1, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2.5, 0.2, 6])
    with col1:
        option = {
            "tooltip": {
                "trigger": 'item'
            },
            "legend": {
                "top": '0%',
                "left": 'center'
            },
            "series": [
                {
                    "name": 'Rating Percentage',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "label": {
                        "show": "false",
                        "position": 'center'
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                    "data": [
                        {"value": 3130, "name": '5 Stars'},
                        {"value": 1036, "name": '4 Stars'},
                        {"value": 355, "name": '3 Stars'},
                        {"value": 120, "name": '2 Stars'},
                        {"value": 76, "name": '1 Star'}
                    ]
                }
            ]
        }

        st_echarts(options=option, key="1")

    with col2:
        st.empty()

    with col3:
        option = {
            "tooltip": {"trigger": 'item'},
            "legend": {
                "right": 'right',
            },
            "series": [{
                "name": 'Clustering Percentage',
                "type": 'pie',
                "radius": ['50%', '75%'],
                "center": ['28%', '50%'],
                "itemStyle": {
                    "borderRadius": "8"
                },
                "label": {
                    "show": "false",
                    "position": 'center'
                },
                "emphasis": {
                    "label": {
                        "show": "False",
                        "fontSize": '10',
                        # "fontWeight": 'bold'
                    }
                },
                "labelLine": {
                    "show": "true"
                },
                "data": [
                    {"value": 1451, "name": 'Entertainment & Activities'},
                    {"value": 690, "name": 'Pool & Beach'},
                    {"value": 615, "name": 'Reaction to negative reviews'},
                    {"value": 614, "name": 'Disagreeing with people reviews'},
                    {"value": 449, "name": 'Breakfast & Room service'},
                    {"value": 372, "name": 'Animation team'},
                    {"value": 162, "name": 'Food'},
                    {"value": 162, "name": 'Interior design & Room decor'},
                    {"value": 111, "name": 'Relaxing atmosphere'},
                    {"value": 91, "name": 'Very positive reviews'}
                ]
            }
            ]
        }
        st_echarts(options=option, key="2")
    st.subheader('\n')

    T2 = '<p style="color:#355070; text-align:left; text-shadow: 0 0 1px #000080; font-size: 30px; font-style:bold">Phenicia Hotel is known for</p>'
    st.markdown(T2, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([3, 0.2, 3])
    with col1:
        option = {
            "series": [
                {
                    "type": 'gauge',
                    "startAngle": 180,
                    "endAngle": 0,
                    "splitNumber": 8,
                    "axisLine": {
                        "lineStyle": {
                            "width": 4,
                            "color": [
                                [0.5, '#FF6E76'],
                                [0.9, '#6D597A'],
                                [1, '#7CFFB2']
                            ]
                        }
                    },
                    "pointer": {
                        "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                        "length": "70%",
                        "width": 10,
                        "offsetCenter": [0, '8%'],
                        "itemStyle": {"color": 'auto'}
                    },
                    "axisTick": {
                        "length": 6,
                        "lineStyle": {
                            "color": 'auto',
                            "width": 1
                        }
                    },
                    "splitLine": {
                        "length": 15,
                        "lineStyle": {
                            "color": 'auto',
                            "width": 5
                        }
                    },
                    "title": {
                        "offsetCenter": [0, '75%'],
                        "fontSize": 20,
                        "color": 'auto',
                    },
                    "data": [
                        {
                            "value": 100,
                            "name": 'Entertainment & Activities \n100% 4/5 Stars Raintings'
                        }
                    ]
                }
            ]
        }

        st_echarts(options=option, key="3")

    with col2:
        st.empty()

    with col3:
        option = {
            "series": [
                {
                    "type": 'gauge',
                    "startAngle": 180,
                    "endAngle": 0,
                    "splitNumber": 8,
                    "axisLine": {
                        "lineStyle": {
                            "width": 4,
                            "color": [
                                [0.5, '#FF6E76'],
                                [0.9, '#6D597A'],
                                [1, '#7CFFB2']
                            ]
                        }
                    },
                    "pointer": {
                        "icon": 'path://M12.8,0.7l12,40.1H0.7L12.8,0.7z',
                        "length": "70%",
                        "width": 10,
                        "offsetCenter": [0, '8%'],
                        "itemStyle": {"color": 'auto'}
                    },
                    "axisTick": {
                        "length": 6,
                        "lineStyle": {
                            "color": 'auto',
                            "width": 1
                        }
                    },
                    "splitLine": {
                        "length": 15,
                        "lineStyle": {
                            "color": 'auto',
                            "width": 5
                        }
                    },
                    "title": {
                        "offsetCenter": [0, '75%'],
                        "fontSize": 20,
                        "color": 'auto'
                    },
                    "data": [
                        {
                            "value": 100,
                            "name": 'Pool & Beach \n100% 4/5 Stars Raintings'
                        }
                    ]
                }
            ]
        }

        st_echarts(options=option, key="4")

    T_Con = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Competitive analysis</h2>'
    st.markdown(T_Con, unsafe_allow_html=True)
    Addresse = '<h3 style="color:#000000; text-align:left; position: absolute; top: -25px; text-shadow: 0 0 1px #000000; font-size: 35px">Hammamet Nabeul Governorate</h3>'
    st.markdown(Addresse, unsafe_allow_html=True)
    df1 = hotels_data.loc[hotels_data['address'] == 'Hammamet_Nabeul_Governorate', :]
    # percentages for Hammamet_Nabeul_Governorate Hotels
    # [[5, 20030], [1, 2371], [2, 2062], [4, 9597], [3, 4391]]
    # [[0, 6696], [6, 6733], [3, 1036], [9, 5088], [5, 2632], [2, 6711], [8, 6384], [7, 755], [4, 1051], [1, 1365]]
    # [[0, 75.97072879330943], [6, 58.82964503193227], [3, 79.72972972972973], [9, 58.19575471698113], [5, 98.59422492401215], [2, 85.03948740873193], [8, 98.41791979949875], [7, 80.26490066225166], [4, 60.60894386298763], [1, 70.62271062271063]]

    T1 = '<p style="color:#355070; text-align:left; position: absolute; top: 25px; text-shadow: 0 0 1px #000080; font-size: 30px; font-style:bold">Ratings & Clustering Percentages</p>'
    st.markdown(T1, unsafe_allow_html=True)

    st.subheader('\n')
    st.subheader('\n')
    st.subheader('\n')
    col1, col2, col3 = st.columns([2.5, 0.2, 6])
    with col1:
        option = {
            "tooltip": {
                "trigger": 'item'
            },
            "legend": {
                "top": '0%',
                "left": 'center'
            },
            "series": [
                {
                    "name": 'Rating Percentage',
                    "type": 'pie',
                    "radius": ['40%', '75%'],
                    "avoidLabelOverlap": "false",
                    "itemStyle": {
                        "borderRadius": "10",
                        "borderColor": '#fff',
                        "borderWidth": "2"
                    },
                    "label": {
                        "show": "false",
                        "position": 'center'
                    },
                    "emphasis": {
                        "label": {
                            "show": "true",
                            "fontSize": '20',
                            "fontWeight": 'bold'
                        }
                    },
                    "labelLine": {
                        "show": "true"
                    },
                    "data": [
                        {"value": 20030, "name": '5 Stars'},
                        {"value": 9597, "name": '4 Stars'},
                        {"value": 4391, "name": '3 Stars'},
                        {"value": 2062, "name": '1 Star'},
                        {"value": 2062, "name": '2 Stars'}
                    ]
                }
            ]
        }

        st_echarts(options=option, key="5")

    with col2:
        st.empty()

    with col3:
        option = {
            "tooltip": {"trigger": 'item'},
            "legend": {
                "right": 'right',
            },
            "series": [{
                "name": 'Clustering Percentage',
                "type": 'pie',
                "radius": ['50%', '75%'],
                "center": ['28%', '50%'],
                "itemStyle": {
                    "borderRadius": "8"
                },
                "label": {
                    "show": "false",
                    "position": 'center'
                },
                "emphasis": {
                    "label": {
                        "show": "False",
                        "fontSize": '10',
                        # "fontWeight": 'bold'
                    }
                },
                "labelLine": {
                    "show": "true"
                },
                "data": [
                    {"value": 6733, "name": 'Disagreeing with people reviews'},
                    {"value": 6711, "name": 'Pool & Beach'},
                    {"value": 6696, "name": 'Breakfast & Room service'},
                    {"value": 6384, "name": 'Entertainment & Activities'},
                    {"value": 5088, "name": 'Reaction to negative reviews'},
                    {"value": 2632, "name": 'Animation team'},
                    {"value": 1365, "name": 'Interior design & Room decor'},
                    {"value": 1051, "name": 'Food'},
                    {"value": 1036, "name": 'Relaxing atmosphere'},
                    {"value": 755, "name": 'Very positive reviews'}
                ]
            }
            ]
        }
        st_echarts(options=option, key="6")

    T = '<h4 style="color:#000000; text-align:left; position: absolute; top: 10px; text-shadow: 0 0 1px #000000; font-size: 20px; font-style:bold">Things to be heightened in marketing campaign</h2>'
    st.markdown(T, unsafe_allow_html=True)
    col1, col2, col3, col4, col5 = st.columns([4, 0.2, 4, 0.2, 4])
    with col1:
        option = {
            "series": [
                {
                    "type": 'gauge',
                    "anchor": {
                        "show": "true",
                        "showAbove": "true",
                        "size": 13,
                        "itemStyle": {"color": '#FAC858'}
                    },

                    "pointer": {
                        "icon": 'path://M2.9,0.7L2.9,0.7c1.4,0,2.6,1.2,2.6,2.6v115c0,1.4-1.2,2.6-2.6,2.6l0,0c-1.4,0-2.6-1.2-2.6-2.6V3.3C0.3,1.9,1.4,0.7,2.9,0.7z',
                        "width": 2.5,
                        "length": '60%',
                        "offsetCenter": [0, '8%']
                    },

                    "progress": {
                        "show": "true",
                        "overlap": "true",
                        "roundCap": "true"
                    },

                    "axisLine": {"roundCap": "true"},
                    "data": [85, 91.8],
                    "detail": {
                        "width": 16,
                        "height": 10,
                        "fontSize": 12,
                        "color": '#fff',
                        "backgroundColor": 'auto',
                        "borderRadius": 4,
                        "formatter": '{value}%'
                    }
                }
            ]
        }
        st_echarts(options=option, key="8")
        T3 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -100px; font-size: 20px; font-style:bold">Pool & Beach</h6>'
        st.markdown(T3, unsafe_allow_html=True)

        R1 = '<p style="color:#B56576; text-align:justify; position: absolute; top: -40px; font-size: 15px">The rating for Phenicias pools was higher than the average rating for pools in that region. </p>'
        st.markdown(R1, unsafe_allow_html=True)

    with col2:
        st.empty()

    with col3:
        option = {
            "series": [
                {
                    "type": 'gauge',
                    "anchor": {
                        "show": "true",
                        "showAbove": "true",
                        "size": 13,
                        "itemStyle": {"color": '#FAC858'}
                    },

                    "pointer": {
                        "icon": 'path://M2.9,0.7L2.9,0.7c1.4,0,2.6,1.2,2.6,2.6v115c0,1.4-1.2,2.6-2.6,2.6l0,0c-1.4,0-2.6-1.2-2.6-2.6V3.3C0.3,1.9,1.4,0.7,2.9,0.7z',
                        "width": 2.5,
                        "length": '60%',
                        "offsetCenter": [0, '8%']
                    },

                    "progress": {
                        "show": "true",
                        "overlap": "true",
                        "roundCap": "true"
                    },

                    "axisLine": {"roundCap": "true"},
                    "data": [79.7, 90.1],

                    "detail": {
                        "width": 16,
                        "height": 10,
                        "fontSize": 12,
                        "color": '#fff',
                        "backgroundColor": 'auto',
                        "borderRadius": 4,
                        "formatter": '{value}%'
                    }
                }
            ]
        }
        st_echarts(options=option, key="9")
        T4 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -100px; font-size: 20px; font-style:bold">Relaxing Atmosphere</h6>'
        st.markdown(T4, unsafe_allow_html=True)
        R2 = '<p style="color:#B56576; text-align:justify; position: absolute; top: -40px; font-size: 15px">The rating for Phenicias atmosphere was higher than the average rating for pools in that region. </p>'
        st.markdown(R2, unsafe_allow_html=True)

    with col4:
        st.empty()

    with col5:
        option = {
            "series": [
                {
                    "type": 'gauge',
                    "anchor": {
                        "show": "true",
                        "showAbove": "true",
                        "size": 13,
                        "itemStyle": {"color": '#FAC858'}
                    },

                    "pointer": {
                        "icon": 'path://M2.9,0.7L2.9,0.7c1.4,0,2.6,1.2,2.6,2.6v115c0,1.4-1.2,2.6-2.6,2.6l0,0c-1.4,0-2.6-1.2-2.6-2.6V3.3C0.3,1.9,1.4,0.7,2.9,0.7z',
                        "width": 2.5,
                        "length": '60%',
                        "offsetCenter": [0, '8%']
                    },

                    "progress": {
                        "show": "true",
                        "overlap": "true",
                        "roundCap": "true"
                    },

                    "axisLine": {"roundCap": "true"},
                    "data": [76, 83.5],

                    "detail": {
                        "width": 16,
                        "height": 10,
                        "fontSize": 12,
                        "color": '#fff',
                        "backgroundColor": 'auto',
                        "borderRadius": 4,
                        "formatter": '{value}%'
                    }
                }
            ]
        }
        st_echarts(options=option, key="10")
        T5 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -100px; font-size: 20px; font-style:bold">Breakfast & Room Service</h6>'
        st.markdown(T5, unsafe_allow_html=True)
        R3 = '<p style="color:#B56576; text-align:justify; position: absolute; top: -40px; font-size: 15px">The rating for Phenicias breakfast and room service was higher than the average rating for pools in that region. </p>'
        st.markdown(R3, unsafe_allow_html=True)

    T = '<h4 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 20px; font-style:bold">Things to be improved</h2>'
    st.markdown(T, unsafe_allow_html=True)

    st.subheader("\n")
    col1, col2, col3 = st.columns([3, 0.2, 3])
    with col1:
        option = {
            "series": [
                {
                    "type": 'gauge',
                    "anchor": {
                        "show": "true",
                        "showAbove": "true",
                        "size": 13,
                        "itemStyle": {"color": '#FAC858'}
                    },

                    "pointer": {
                        "icon": 'path://M2.9,0.7L2.9,0.7c1.4,0,2.6,1.2,2.6,2.6v115c0,1.4-1.2,2.6-2.6,2.6l0,0c-1.4,0-2.6-1.2-2.6-2.6V3.3C0.3,1.9,1.4,0.7,2.9,0.7z',
                        "width": 2.5,
                        "length": '60%',
                        "offsetCenter": [0, '8%']
                    },

                    "progress": {
                        "show": "true",
                        "overlap": "true",
                        "roundCap": "true"
                    },

                    "axisLine": {"roundCap": "true"},
                    "data": [60.6, 77.8],
                    "detail": {
                        "width": 16,
                        "height": 10,
                        "fontSize": 12,
                        "color": '#fff',
                        "backgroundColor": 'auto',
                        "borderRadius": 4,
                        "formatter": '{value}%'
                    }
                }
            ]
        }
        st_echarts(options=option, key="11")
        T6 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -70px; font-size: 20px; font-style:bold">Food</h6>'
        st.markdown(T6, unsafe_allow_html=True)
        R4 = '<p style="color:#B56576; text-align:justify; position: absolute; top: -40px; font-size: 15px">The food is relatively poorly rated at Phenicia Hotel and in Hamamet in general. </p>'
        st.markdown(R4, unsafe_allow_html=True)

    with col2:
        st.empty()

    with col3:
        option = {
            "series": [
                {
                    "type": 'gauge',
                    "anchor": {
                        "show": "true",
                        "showAbove": "true",
                        "size": 13,
                        "itemStyle": {"color": '#FAC858'}
                    },

                    "pointer": {
                        "icon": 'path://M2.9,0.7L2.9,0.7c1.4,0,2.6,1.2,2.6,2.6v115c0,1.4-1.2,2.6-2.6,2.6l0,0c-1.4,0-2.6-1.2-2.6-2.6V3.3C0.3,1.9,1.4,0.7,2.9,0.7z',
                        "width": 2.5,
                        "length": '60%',
                        "offsetCenter": [0, '8%']
                    },

                    "progress": {
                        "show": "true",
                        "overlap": "true",
                        "roundCap": "true"
                    },

                    "axisLine": {"roundCap": "true"},
                    "data": [70.6, 82],

                    "detail": {
                        "width": 16,
                        "height": 10,
                        "fontSize": 12,
                        "color": '#fff',
                        "backgroundColor": 'auto',
                        "borderRadius": 4,
                        "formatter": '{value}%'
                    }
                }
            ]
        }
        st_echarts(options=option, key="12")
        T7 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -70px; font-size: 20px; font-style:bold">Interior Design & Room Decor</h6>'
        st.markdown(T7, unsafe_allow_html=True)
        R5 = '<p style="color:#B56576; text-align:justify; position: absolute; top: -40px; font-size: 15px">Interior Design and Room Decoration are relatively poorly rated at the Phenicia Hotel and in Hamamet in general. </p>'
        st.markdown(R5, unsafe_allow_html=True)
















nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('vader_lexicon')
st.set_page_config(layout="wide")

loaded_model = pickle.load(open('pages/finalized_model.sav', 'rb'))
def convert_to_df(sentiment):
    sentiment_dict = {'polarity': sentiment.polarity, 'subjectivity': sentiment.subjectivity}
    sentiment_df = pd.DataFrame(sentiment_dict.items(), columns=['metric', 'value'])
    return sentiment_df


def analyze_token_sentiment(docx):
    analyzer = SentimentIntensityAnalyzer()
    pos_list = []
    neg_list = []
    neu_list = []
    for i in docx.split():
        res = analyzer.polarity_scores(i)['compound']
        if res > 0.1:
            pos_list.append(i)
            pos_list.append(res)

        elif res <= -0.1:
            neg_list.append(i)
            neg_list.append(res)
        else:
            neu_list.append(i)

    result = {'positives': pos_list, 'negatives': neg_list, 'neutral': neu_list}
    return result

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
    st.markdown("""
    <style>
    .big-font {
        font-size:16px !important;
        font-family: "Monaco", "Monaco", monospace;

    }
    </style>
    """, unsafe_allow_html=True)
    col1, col2, col3, col4,col5 = st.columns([1, 2.5, 2.5, 2.5,1])
    col1.write("")
    col2.markdown("<h3 style='text-align: center; color: #B56576;'>Hotels And Tourism</h3>", unsafe_allow_html=True)
    exp= col2.expander('show more')
    exp.markdown('<p class="big-font"> description here </p>',unsafe_allow_html=True)

    col3.markdown("<h3 style='text-align: center; color: #B56576;'>Food Retail Stores</h3>", unsafe_allow_html=True)
    expf= col3.expander('show more')
    expf.markdown('<p class="big-font"> Consider a well-established company operating in the retail food sector. <br> '
                 'Presently they have around several hundred thousands of registered customers and serve almost one '
                 'million consumers a year. <br> They sell products from 5 major categories: Dairy, rare meat '
                 'products, exotic fruits, specially prepared fish and sweet products. These can further be divided '
                 'into gold and regular products. <br> The customers can order and acquire products through 3 sales '
                 'channels: physical stores, catalogs and company’s website.  <br> Globally, the company had solid '
                 'revenues and a healthy bottom line in the past 3 years, but the profit growth perspectives for the '
                 'next 3 years are not promising… <br> For this reason, several strategic initiatives are being '
                 'considered to invert this situation. <br> One is to improve the  performance of marketing '
                 'activities, with a special focus on marketing campaigns. </p>', unsafe_allow_html=True)
    col4.markdown("<h3 style='text-align: center; color: #B56576;'>Banking Business</h3>", unsafe_allow_html=True)
    expb=col4.expander('show more')
    expb.write("description here")
    col5.write("")
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

def food(st,**data):
    st.sidebar.header('Input Parameters')
    def user_input_features():
        Age = st.sidebar.slider('age', 0, 100, 30)
        Education = st.sidebar.selectbox(
            'Education:',
            ('Graduation','PhD','Master','Basic','2n Cycle'))
        Marital_Status = st.sidebar.selectbox('Marital Status:', ('Married','Together','Alone','Widow'))
        Income= st.sidebar.number_input('Income:')
        Kidhome = st.sidebar.slider('Number of kids home', 0, 10, 1)
        Teenhome = st.sidebar.slider('Number of teens home', 0, 10, 1)
        Dt_Customer= st.sidebar.date_input('When did they become a registered customer')
        Recency= st.sidebar.number_input('Recency:',0,100, step=1)
        MntDairy= st.sidebar.number_input('Amount of Dairy purchased',0,1200, step=1)
        MntFruits= st.sidebar.number_input('Amount of fruits purchased:',0,200, step=1)
        MntMeatProducts= st.sidebar.number_input('Amount of meat products purchased:',0,2000, step=1)
        MntFishProducts= st.sidebar.number_input('Amount of fish products purchased:',0,2000, step=1)
        MntSweetProducts=st.sidebar.number_input('Amount of sweet products purchased:',0,2000, step=1)
        MntGold= st.sidebar.number_input('Amount of luxury products purchased:',0,2000, step=1)
        NumDealsPurchases= st.sidebar.number_input('Number of deals purchases', step=1)
        NumWebPurchases= st.sidebar.number_input('Number of web purchases:', step=1)
        NumCatalogPurchases= st.sidebar.number_input('Number of catalog purchases', step=1)
        NumStorePurchases= st.sidebar.number_input('Number of store purchases', step=1)
        NumWebVisitsMonth= st.sidebar.number_input('Number of web visits per month',step=1)
        AcceptedCmp1= st.sidebar.slider('Accepted campaign 1:',0,1,0)
        AcceptedCmp2= st.sidebar.slider('Accepted campaign 2:',0,1,0)
        AcceptedCmp3= st.sidebar.slider('Accepted campaign 3:',0,1,0)
        AcceptedCmp4= st.sidebar.slider('Accepted campaign 4:',0,1,0)
        AcceptedCmp5= st.sidebar.slider('Accepted campaign 5:',0,1,0)

        data = {'Age': Age,
                'Education': Education,
                'Marital_Status': Marital_Status,
                'Income': Income,
                'Kidhome': Kidhome,
                'Teenhome':Teenhome,
                'Dt_Customer': Dt_Customer,
                'Year_Customer': Dt_Customer.year,
                'Recency':Recency,
                'MntDairy':MntDairy,
                'MntFruits': MntFruits,
                'MntMeatProducts': MntMeatProducts,
                'MntFishProducts': MntFishProducts,
                'MntSweetProducts': MntSweetProducts,
                'MntGold': MntGold,
                'NumDealsPurchases': NumDealsPurchases,
                'NumWebPurchases': NumWebPurchases,
                'NumCatalogPurchases': NumCatalogPurchases,
                'NumStorePurchases': NumStorePurchases,
                'NumWebVisitsMonth': NumWebVisitsMonth,
                'AcceptedCmp3': AcceptedCmp3== True,
                'AcceptedCmp4': AcceptedCmp4== True,
                'AcceptedCmp5': AcceptedCmp5==True ,
                'AcceptedCmp1': AcceptedCmp1==True,
                'AcceptedCmp2': AcceptedCmp2== True,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    data = user_input_features()
    data.insert(7, 'NumChildrenhome', data["Kidhome"]+ data["Teenhome"])
    data.insert(15, 'Spendings', data["MntDairy"] + data["MntFruits"] + data["MntMeatProducts"]+ data["MntFishProducts"]+data["MntSweetProducts"])
    data.insert(8, 'Haskids', data["NumChildrenhome"]>0)
    data.insert(3, 'Years_Education', np.zeros  )
    data["Years_Education"]=data['Education'].replace(['Basic', 'Graduation', 'PhD', 'Master', '2n Cycle'],
                            [6, 13, 21, 18, 9], inplace=False)
    data.drop('Education',axis=1,inplace=True)
    data["Income_status"] = np.nan
    lst = [data]

    for col in lst:
        col.loc[(col["Income"] >= 0) & (col["Income"] <= 9600), "Income_status"] = "low"
        col.loc[(col["Income"] > 9600) & (col["Income"] <= 36000), "Income_status"] = "middle"
        col.loc[col["Income"] > 36000, "Income_status"] = "high"


    st.subheader('User Input parameters')
    st.write(data)


    result= loaded_model.predict(data)
    resultproba = loaded_model.predict_proba(data)

    st.subheader('Class labels and their corresponding index number')
    st.write(['Refused Marketing Campaign', 'Accepted marketing Campaign'])

    st.subheader('Prediction results in probability')
    st.write(resultproba)
    print(result)
def sentiment(st,**data):
    st.title("Sentiment Analysis NLP App")
    st.subheader("Streamlit Projects")

    st.subheader("Client Feedback Sentiment Analysis")
    with st.form(key='nlpForm'):
        raw_text = st.text_area("Enter Text Here")
        submit_button = st.form_submit_button(label='Analyze')

    # layout
    col1, col2 = st.columns([1 , 1])
    if submit_button:

        with col1:
            st.info("Results")
            sentiment = TextBlob(raw_text).sentiment

            # Emoji
            if sentiment.polarity > 0:
                st.markdown("Sentiment:: Positive :smiley: ")
            elif sentiment.polarity < 0:
                st.markdown("Sentiment:: Negative :angry: ")
            else:
                st.markdown("Sentiment:: Neutral 😐 ")

            # Dataframe
            result_df = convert_to_df(sentiment)
            st.dataframe(result_df)

            # Visualization
            c = alt.Chart(result_df).mark_bar().encode(
                x='metric',
                y='value',
                color='metric')
            st.altair_chart(c, use_container_width=True)

        with col2:
            st.info("Token Sentiment")

            token_sentiments = analyze_token_sentiment(raw_text)
            st.write(token_sentiments)


app = MultiPage()
app.st = st

app.add_app("Home Page", home, initial_page=True)
app.add_app("1", food)
app.add_app("2", sentiment)
app.add_app("3",hotel)
app.add_app("4", home)

app.run()
