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

from pyecharts import options as opts
from pyecharts.charts import Bar
from streamlit_echarts import st_pyecharts

st.set_page_config(layout="wide")
Titre_principal = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Phenicia Hotel</h2>'
st.markdown(Titre_principal, unsafe_allow_html=True)

st.subheader('\n')

DATA_URL = "./Hammamet_DataBase.csv"


@st.cache(
    persist=True)  # ( If you have a different use case where the data does not change so very often, you can simply use this)
def load_data():
    data = pd.read_csv(DATA_URL)
    return data


df1 = load_data()

df = df1.loc[df1['hotel'] == 'Phenicia_Hotel', :]
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
        st.info("Number of reviews for Phenicia Hotel: **_4717_** reviews")
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

col1, col2, col3 = st.columns([3.5, 0.2, 6])
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

T_Con = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Competitive analysis</h2>'
st.markdown(T_Con, unsafe_allow_html=True)
Addresse = '<h3 style="color:#000000; text-align:left; position: absolute; top: -25px; text-shadow: 0 0 1px #000000; font-size: 35px">Hammamet Nabeul Governorate</h3>'
st.markdown(Addresse, unsafe_allow_html=True)
# percentages for Hammamet_Nabeul_Governorate Hotels
# [[5, 20030], [1, 2371], [2, 2062], [4, 9597], [3, 4391]]
# [[0, 6696], [6, 6733], [3, 1036], [9, 5088], [5, 2632], [2, 6711], [8, 6384], [7, 755], [4, 1051], [1, 1365]]
# [[0, 75.97072879330943], [6, 58.82964503193227], [3, 79.72972972972973], [9, 58.19575471698113], [5, 98.59422492401215], [2, 85.03948740873193], [8, 98.41791979949875], [7, 80.26490066225166], [4, 60.60894386298763], [1, 70.62271062271063]]

T1 = '<p style="color:#355070; text-align:left; position: absolute; top: 25px; text-shadow: 0 0 1px #000080; font-size: 30px; font-style:bold">Ratings & Clustering Percentages</p>'
st.markdown(T1, unsafe_allow_html=True)

st.subheader('\n')
st.subheader('\n')
st.subheader('\n')
col1, col2, col3 = st.columns([3.5, 0.2, 6])
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

clusters_names = ["Breakfast and Room service", "Interior design and Room decoration", "Pool and Beach",
                  "Relaxing atmosphere", "Food", "Animation team", "Disagreeing with people reviews",
                  "Very positive reviews", "Entertainment and Activities", "Reaction to negative reviews"]
a = []
count_clusters = []
for i in range(len(df.cluster_assignment.unique())):
    count_clusters.append(
        [df.cluster_assignment.unique()[i], (df['cluster_assignment'] == df.cluster_assignment.unique()[i]).sum()])
sorted(count_clusters, key=itemgetter(1), reverse=True)
print(count_clusters)
for i in range(len(df.cluster_assignment.unique())):
    a.append({"value": str(count_clusters[i][1]), "name": clusters_names[count_clusters[i][0]]})
print('ùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùùù')
t = json.dumps(a)
print(t)

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
            "data": t
        }
        ]
    }
    st_echarts(options=option, key="6")

T = '<h4 style="color:#000000; text-align:left; position: absolute; top: 0px; text-shadow: 0 0 1px #000000; font-size: 20px; font-style:bold">Things to be highlighted in marketing campaign</h2>'
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
    T3 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -80px; font-size: 20px; font-style:bold">Pool & Beach</h6>'
    st.markdown(T3, unsafe_allow_html=True)

    R1 = '<p style="color:#B56576; text-align:justify; position: absolute; top: -55px; font-size: 15px">The rating for Phenicias pools was higher than the average rating for pools in that region. </p>'
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
    T4 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -80px; font-size: 20px; font-style:bold">Relaxing Atmosphere</h6>'
    st.markdown(T4, unsafe_allow_html=True)
    R2 = '<p style="color:#B56576; text-align:justify; position: absolute; top: -55px; font-size: 15px">The rating for Phenicias atmosphere was higher than the average rating for pools in that region. </p>'
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
    T5 = '<h6 style="color:#000000; text-align:center; position: absolute; top: -80px; font-size: 20px; font-style:bold">Breakfast & Room Service</h6>'
    st.markdown(T5, unsafe_allow_html=True)
    R3 = '<p style="color:#B56576; text-align:justify; position: absolute; top: -55px; font-size: 15px">The rating for Phenicias breakfast and room service was higher than the average rating for pools in that region. </p>'
    st.markdown(R3, unsafe_allow_html=True)

st.subheader('\n')
st.subheader('\n')
T = '<h4 style="color:#000000; text-align:left; position: absolute; top: 10px; text-shadow: 0 0 1px #000000; font-size: 20px; font-style:bold">Things that has to be improved given relatively poor ratings:</h2>'
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

# Extract_Mean_Rating_per_date for Phenicia_Hotel
df['date'] = df['date'].apply(lambda x: datetime.fromisoformat(x + ' 00:00:00'))
base = df.groupby('date').aggregate({"rating": [list, "count", "mean"]}).reset_index()
base["count_rating_per_date"] = base["rating"]["count"]
base["mean_rating_per_date"] = base["rating"]["mean"]
base.drop(["rating"], axis=1, inplace=True)
base.set_index('date', inplace=True)
base = base.sort_index()
Ratings_monthly = base.resample('M').sum()

st.subheader('\n')
st.subheader('\n')
T8 = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Direct analysis</h2>'
st.markdown(T8, unsafe_allow_html=True)
T9 = '<h4 style="color:#000000; text-align:left; position: absolute; top: -25px; text-shadow: 0 0 1px #000000; font-size: 30px">Count Rating Per Date</h4>'
st.markdown(T9, unsafe_allow_html=True)

st.subheader('\n')
st.subheader('\n')
st.line_chart(Ratings_monthly['count_rating_per_date'])

loaded_model = pickle.load(open('model.sav', 'rb'))


def get_n_last_months(df, series_name, n_months):
    return df[series_name][-(n_months):]


def get_keras_format_series(series):
    series = np.array(series)
    return series.reshape(series.shape[0], series.shape[1], 1)


def get_train_test_data(df, series_name, series_months, input_months,
                        test_months, sample_gap=2):
    forecast_series = get_n_last_months(df, series_name,
                                        series_months).values  # reducing our forecast series to last n months

    train = forecast_series[:-test_months]  # training data is remaining months until amount of test_months
    test = forecast_series[-test_months:]  # test data is the remaining test_months

    train_X, train_y = [], []

    # range 0 through # of train samples - input_months by sample_gap.
    # This is to create many samples with corresponding
    for i in range(0, train.shape[0] - input_months, sample_gap):
        train_X.append(train[i:i + input_months])  # each training sample is of length input months
        train_y.append(train[i + input_months])  # each y is just the next step after training sample

    train_X = get_keras_format_series(train_X)  # format our new training set to keras format
    train_y = np.array(train_y)  # make sure y is an array to work properly with keras

    # The set that we had held out for testing (must be same length as original train input)
    test_X_init = test[:input_months]
    test_y = test[input_months:]  # test_y is remaining values from test set

    return train_X, test_X_init, train_y, test_y


def predict(X_init, n_steps):
    X_init = X_init.copy().reshape(1, -1, 1)
    preds = []
    for _ in range(n_steps):
        pred = loaded_model.predict(X_init)
        pred = pred[0][0]
        preds.append(pred)
        X_init[:, :-1, :] = X_init[:, 1:, :]  # replace first 11 values with 2nd through 12th
        X_init[:, -1, :] = pred  # replace 12th value with prediction
    return preds


def predict_and_plot(X_init, y, title):
    y_preds = predict(test_X_init, n_steps=len(y))  # predict through length of y
    # Below ranges are to set x-axes
    start_range = range(1, test_X_init.shape[0] + 1)  # starting at one through to length of test_X_init to plot X_init
    predict_range = range(test_X_init.shape[0],
                          test_months)  # predict range is going to be from end of X_init to length of test_hours

    fig = px.line(
        x=start_range, y=test_X_init, title="X_init")
    st.plotly_chart(fig)
    fig = px.line(
        x=predict_range, y=test_y, title="Test")
    st.plotly_chart(fig)
    fig = px.line(
        x=predict_range, y=y_preds, title="Actual Preds")
    st.plotly_chart(fig)


loaded_model = pickle.load(open("./finalized_model.sav", 'rb'))
series_months = 194
input_months = 12
test_months = 48
train_X, test_X_init, train_y, test_y = \
    (get_train_test_data(Ratings_monthly, 'count_rating_per_date', series_months,
                         input_months, test_months))
predict_and_plot(test_X_init, test_y, 'Reviews counts Series: Test Data and LSTM Predictions')