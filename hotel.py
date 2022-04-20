from cProfile import label
from tkinter import PIESLICE
from streamlit_echarts import st_echarts
import streamlit as st
import pandas as pd
import numpy as np
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
def hotel():
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