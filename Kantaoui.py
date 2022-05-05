from cProfile import label
import json
import pickle
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

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN, LSTM, Activation, Dropout

from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts
st.set_page_config(layout="wide")
Titre_principal = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Palmyra Aquapark Kantaoui Hotel</h2>'
st.markdown(Titre_principal, unsafe_allow_html=True)

st.subheader('\n')

DATA_URL="./Kantaoui_DataBase.csv"
@st.cache(persist=True)  #( If you have a different use case where the data does not change so very often, you can simply use this)
def load_data():
    data=pd.read_csv(DATA_URL)
    return data
df1=load_data()

df=df1.loc[df1['hotel']=='Palmyra_Aquapark_Kantaoui',:]
a=len(df)
print(a)

#Percentages for Palmyra_Aquapark_Kantaoui Hotel
#stars=[{'value': 4, 'name': 463}, {'value': 5, 'name': 410}, {'value': 3, 'name': 306}, {'value': 1, 'name': 288}, {'value': 2, 'name': 194}]
#clusters=[{'value': 6, 'name': 488}, {'value': 9, 'name': 328}, {'value': 2, 'name': 232}, {'value': 4, 'name': 172}, {'value': 8, 'name': 146}, {'value': 0, 'name': 145}, {'value': 1, 'name': 53}, {'value': 5, 'name': 43}, {'value': 3, 'name': 32}, {'value': 7, 'name': 22}]
##["Breakfast and Room service", "Interior design and Room decoration", "Pool and Beach", "Relaxing atmosphere", "Food", "Animation team", "Disagreeing with people reviews", "Very positive reviews", "Entertainment and Activities", "Reaction to negative reviews"]
#Top favourite clusters=[[0, 48.275862068965516], [3, 50.0], [6, 42.00819672131148], [9, 36.28048780487805], [4, 51.162790697674424], [2, 71.98275862068965], [8, 90.41095890410958], [7, 50.0], [1, 43.39622641509434], [5, 97.67441860465115]]
#affichage de a


number = st.sidebar.number_input('Insert the number of desired predicted months', min_value=1, max_value=100, value=36, step=1)
#bech naamel affichage lel wetla eli bahdheha (infos)
col1, col2, col3, col4,col5, col6, col7=st.columns([0.3, 2, 0.3, 2, 0.3, 2, 0.3 ])

with col1:
    st.empty()
with col2:
    with st.container():
        st.info("**_70.9%_ of travelers say online content influences their choice of where to stay. (RMS)**")
with col3:
    st.empty()
with col4:
    with st.container():
        st.info("**_52%_ of individuals would never book a hotel that had zero reviews. (TripAdvisor)**")
with col5:
    st.empty()
with col6:
    with st.container():
        st.info("**_65%_ of people gain their travel inspiration from online searches. (Google)**")
with col7:
    st.empty()

st.subheader('\n')
st.subheader('\n')

T1 = '<p style="color:#355070; text-align:left; text-shadow: 0 0 1px #000080; font-size: 30px; font-style:bold">Ratings & Clustering Percentages</p>'
st.markdown(T1, unsafe_allow_html=True)


col1, col2, col3=st.columns([3, 0.2, 6])
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
            "avoidLabelOverlap": "true",
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
                "show": "false"
            },
            "data": [
                {'value': 463, 'name': '4 Stars'}, 
                {'value': 410, 'name': '5 Stars'}, 
                {'value': 306, 'name': '3 Stars'}, 
                {'value': 288, 'name': '1 Star'},
                {'value': 194, 'name': '2 Stars'}
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
                    #"fontWeight": 'bold'
                    }
                },
            "labelLine": {
                "show": "true"
            },
            "data": [
                {"value": 488, "name": 'Disagreeing with people reviews'},
                {"value": 328, "name": 'Reaction to negative reviews'},
                {"value": 232, "name": 'Pool and Beach'},
                {"value": 172, "name": 'Food'},
                {"value": 146, "name": 'Entertainment and Activities'},
                {"value": 145, "name": 'Breakfast and Room service'},
                {"value": 53, "name": 'Interior design and Room decoration'},
                {"value": 43, "name": 'Animation team'},
                {"value": 32, "name": 'Relaxing atmosphere'},
                {"value": 22, "name": 'Very positive reviews'}
            ]
        }
    ]
}               
    st_echarts(options=option, key="2")
st.subheader('\n')


T_Con = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Competitive analysis</h2>'
st.markdown(T_Con, unsafe_allow_html=True)
Addresse = '<h3 style="color:#000000; text-align:left; position: absolute; top: -25px; text-shadow: 0 0 1px #000000; font-size: 35px">Port El Kantaoui Sousse Governorate</h3>'
st.markdown(Addresse, unsafe_allow_html=True)

#percentages for Kantaoui Hotels
#[[5, 9055], [4, 6202], [3, 2930], [1, 1439], [2, 1408]]
#[[6, 4577], [2, 4220], [0, 3697], [9, 3137], [8, 2320], [5, 828], [1, 784], [4, 639], [7, 412], [3, 420]]  
#[[0, 74.79037057073302], [6, 58.44439589250601], [8, 97.62931034482759], [2, 84.02843601895734], [5, 98.30917874396135], [9, 53.26745298055467], [3, 73.57142857142858], [1, 68.62244897959184], [7, 79.36893203883496], [4, 54.30359937402191]]

T1 = '<p style="color:#355070; text-align:left; position: absolute; top: 25px; text-shadow: 0 0 1px #000080; font-size: 30px; font-style:bold">Ratings & Clustering Percentages</p>'
st.markdown(T1, unsafe_allow_html=True)

st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
col1, col2, col3=st.columns([3, 0.2, 6])
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
                {"value": 9055, "name": '5 Stars'},
                {"value": 6202, "name": '4 Stars'},
                {"value": 2930, "name": '3 Stars'},
                {"value": 1439, "name": '1 Star'},
                {"value": 1408, "name": '2 Stars'}
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
                    #"fontWeight": 'bold'
                    }
                },
            "labelLine": {
                "show": "true"
            },
            "data": [
                {"value": 4577, "name": 'Disagreeing with people reviews'},
                {"value": 4220, "name": 'Pool & Beach'},
                {"value": 3697, "name": 'Breakfast & Room service'},
                {"value": 3137, "name": 'Reaction to negative reviews'},
                {"value": 2320, "name": 'Entertainment & Activities'},
                {"value": 828, "name": 'Animation team'},
                {"value": 784, "name": 'Interior design and Room decoration'},
                {"value": 639, "name": 'Food'},
                {"value": 412, "name": 'Very positive reviews'},
                {"value": 420, "name": 'Relaxing atmosphere'}
            ]
        }
    ]
}               
    st_echarts(options=option, key="6")

T = '<h4 style="color:#000000; text-align:left; position: absolute; top: 0px; text-shadow: 0 0 1px #000000; font-size: 20px; font-style:bold">Things to be highlighted in marketing campaign</h2>'
st.markdown(T, unsafe_allow_html=True)
col1, col2, col3, col4, col5=st.columns([4, 0.2, 4, 0.2, 4])
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
        "data": [98.3,97.7],
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
    T3 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -80px; font-size: 20px; font-style:bold">Animation team</h6>'
    st.markdown(T3, unsafe_allow_html=True)
    P= '<p style="color:#000000; text-align:left; position: absolute; line-height:1em; top: -65px; font-size: 17px; font-style:bold">Percentage of 4 and 5 star ratings for the hotel: 97.7% <br>Percentage of 4 and 5 star ratings for all hotels in Kantaoui: 98.3%</p>'
    st.markdown(P, unsafe_allow_html=True)

    R1 = '<p style="color:#B56576; text-align:justify; position: absolute; top: 0px; font-size: 15px">Animation team ratings were excellent, both for the hotel and for the government in general. Nevertheless, less than 4% of guests noticed this excellent service. \nCustomers may be more concerned about the hotel s many poor services, hence the need to improve the poorly rated services and highlight the highly rated services.</p>'
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
        "data": [97.6,90.4],
        
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
    T4 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -80px; font-size: 20px; font-style:bold">Entertainment & Activities</h6>'
    st.markdown(T4, unsafe_allow_html=True)
    P= '<p style="color:#000000; text-align:left; position: absolute; line-height:1em; top: -65px; font-size: 17px; font-style:bold">Percentage of 4 and 5 star ratings for the hotel: 90.4% <br>Percentage of 4 and 5 star ratings for all hotels in Kantaoui: 97.6%</p>'
    st.markdown(P, unsafe_allow_html=True)
    R2 = '<p style="color:#B56576; text-align:justify; position: absolute; top: 0px; font-size: 15px">Ratings for entertainment and activities were also excellent, and more clients noticed these services (More then 12%).</p>'
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
        "data": [84,72],
        
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
    T5 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -80px; font-size: 20px; font-style:bold">Pool & Beach</h6>'
    st.markdown(T5, unsafe_allow_html=True)
    P= '<p style="color:#000000; text-align:left; position: absolute; line-height:1em; top: -65px; font-size: 17px; font-style:bold">Percentage of 4 and 5 star ratings for the hotel: 72% <br>Percentage of 4 and 5 star ratings for all hotels in Kantaoui: 84%</p>'
    st.markdown(P, unsafe_allow_html=True)
    R3 = '<p style="color:#B56576; text-align:justify; position: absolute; top: 0px; font-size: 15px">The most appreciated thing in this hotel and government, is its pool and beach, and with high rates, (More than 20% of customers mentioned the pools and beaches). Thus, the hotel is still underrated compared to competing hotels in the same area. </p>'
    st.markdown(R3, unsafe_allow_html=True)

st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
T = '<h4 style="color:#000000; text-align:left; position: absolute; top: 10px; text-shadow: 0 0 1px #000000; font-size: 20px; font-style:bold">Things that has to be improved given relatively poor ratings:</h2>'
st.markdown(T, unsafe_allow_html=True)

st.subheader("\n")
col1, col2, col3=st.columns([3, 0.2, 3])
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
        "data": [74.8,48.2],
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
    T6 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -70px; font-size: 20px; font-style:bold">Breakfast & Room service</h6>'
    st.markdown(T6, unsafe_allow_html=True)
    P= '<p style="color:#000000; text-align:left; position: absolute; line-height:1em; top: -60px; font-size: 17px; font-style:bold">Percentage of 4 and 5 star ratings for the hotel: 48.2% <br>Percentage of 4 and 5 star ratings for all hotels in Kantaoui: 74.8%</p>'
    st.markdown(P, unsafe_allow_html=True)
    R4 = '<p style="color:#B56576; text-align:justify; position: absolute; top: 0px; font-size: 15px">Breakfast and Room service are poorly rated in this hotel, we can actually notice the huge difference between the ratings of this hotel and those of competing hotels in the same area, knowing that about 20% of the guests have expressed criticism on this topic. </p>'
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
        "data": [54.3,51.1],
        
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
    T7 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -70px; font-size: 20px; font-style:bold">Food</h6>'
    st.markdown(T7, unsafe_allow_html=True)
    P= '<p style="color:#000000; text-align:left; position: absolute; line-height:1em; top: -60px; font-size: 17px; font-style:bold">Percentage of 4 and 5 star ratings for the hotel: 51.1% <br>Percentage of 4 and 5 star ratings for all hotels in Kantaoui: 54.3%</p>'
    st.markdown(P, unsafe_allow_html=True)
    R5 = '<p style="color:#B56576; text-align:justify; position: absolute; top: 0px; font-size: 15px">The Food in general is rated poorly in this hotel, and in all El Kantaoui Sousse hotels.</p>'
    st.markdown(R5, unsafe_allow_html=True)

st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
col1, col2, col3=st.columns([3, 0.2, 3])
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
        "data": [68.6,43.4],
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
    st_echarts(options=option, key="13")
    T6 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -70px; font-size: 20px; font-style:bold">Interior design & Room decoration</h6>'
    st.markdown(T6, unsafe_allow_html=True)
    P= '<p style="color:#000000; text-align:left; position: absolute; line-height:1em; top: -60px; font-size: 17px; font-style:bold">Percentage of 4 and 5 star ratings for the hotel: 43.4% <br>Percentage of 4 and 5 star ratings for all hotels in Kantaoui: 68.6%</p>'
    st.markdown(P, unsafe_allow_html=True)

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
        "data": [73.6,50],
        
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
    st_echarts(options=option, key="14")
    T7 = '<h6 style="color:#000000; text-align:right; position: absolute; top: -70px; font-size: 20px; font-style:bold">Relaxing atmosphere</h6>'
    st.markdown(T7, unsafe_allow_html=True)
    P= '<p style="color:#000000; text-align:left; position: absolute; line-height:1em; top: -60px; font-size: 17px; font-style:bold">Percentage of 4 and 5 star ratings for the hotel: 50.0% <br>Percentage of 4 and 5 star ratings for all hotels in Kantaoui: 73.6%</p>'
    st.markdown(P, unsafe_allow_html=True)

col1, col2, col3=st.columns([1, 3, 1])
with col1:
    st.empty()
with col2:
    R5 = '<p style="color:#B56576; text-align:justify; position: absolute; top: 5px; font-size: 15px">Interior design, Room decoration and General Atmosphere were neglected by clients. <br>In fact, less than 6% of clients commented on these topics, and they were not satisfied with these types of services. </p>'
    st.markdown(R5, unsafe_allow_html=True)
with col3:
    st.empty()

#Extract_Mean_Rating_per_date for Phenicia_Hotel
df['date']=df['date'].apply(lambda x: datetime.fromisoformat(x+' 00:00:00'))
base = df.groupby('date').aggregate({"rating": [list, "count","mean"]}).reset_index()
base["count_rating_per_date"] = base["rating"]["count"]
base["mean_rating_per_date"] = base["rating"]["mean"]
base.drop(["rating"], axis=1, inplace=True)
base.set_index('date', inplace=True)
base=base.sort_index()
Ratings_monthly = base.resample('M').sum()

st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
T8 = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Direct analysis</h2>'
st.markdown(T8, unsafe_allow_html=True)
T9 = '<h4 style="color:#000000; text-align:left; position: absolute; top: -25px; text-shadow: 0 0 1px #000000; font-size: 30px">Count Rating Per Date</h4>'
st.markdown(T9, unsafe_allow_html=True)

st.subheader('\n')
st.subheader('\n')
st.line_chart(Ratings_monthly['count_rating_per_date'])
