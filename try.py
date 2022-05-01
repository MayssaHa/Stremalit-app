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

def hammamet(st, **data):
    from cProfile import label
    import json
    from tkinter import PIESLICE
    from streamlit_echarts import st_echarts
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
                            "show": "false",
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
        start_range = range(1,
                            test_X_init.shape[0] + 1)  # starting at one through to length of test_X_init to plot X_init
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

    loaded_model = pickle.load(open("model.sav", 'rb'))
    series_months = 194
    input_months = 12
    test_months = 48
    train_X, test_X_init, train_y, test_y = \
        (get_train_test_data(Ratings_monthly, 'count_rating_per_date', series_months,
                             input_months, test_months))
    predict_and_plot(test_X_init, test_y, 'Reviews counts Series: Test Data and LSTM Predictions')

def hotel(st, **data):
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

    st.title("Tunisia Hotels")
    st.header("WELCOME")
    # new_title = '<p style="color:Blue; font-size: 28px;">Please select an hotel from the list to get **an overview of the customer feedbacks**, **the variation of these feedbacks over time**, as well as **the variation of the number of customers visiting this hotel over time**.</p>'
    # st.markdown(new_title, unsafe_allow_html=True)
    st.markdown(
        "Please select an hotel from the list to get **an overview of the customer feedbacks**, **the variation of these feedbacks over time**, as well as **the variation of the number of customers visiting this hotel over time**.")

    DATA_URL = "./hotel_clusters.csv"

    @st.cache(
        persist=True)  # ( If you have a different use case where the data does not change so very often, you can simply use this)
    def load_data():
        data = pd.read_csv(DATA_URL)
        return data

    hotels = load_data()

    st.sidebar.checkbox("Show Analysis by Address", True, key=1)

    select = st.sidebar.selectbox('Select an Address', hotels['address'].unique())
    # get the address selected in the selectbox
    address_data = hotels[hotels['address'] == select]
    print(address_data['hotel'].value_counts())

    select2 = st.sidebar.selectbox('Select an Hotel', address_data['hotel'].unique())
    # select_cluster = st.sidebar.radio("Topic", ('Cluster0', 'Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6', 'Cluster7', 'Cluster8', 'Cluster9'))

    hotel_data = address_data.loc[address_data['hotel'] == select2, :]
    hotel_data.drop(
        columns=['count_rating_per_date', 'tf_idf', 'distances', 'mean_rating_per_date', 'count_rating_per_year',
                 'mean_rating_per_year', 'avg_rating'], inplace=True)
    count = (address_data['hotel'] == select2).sum()

    st.subheader('\n')
    st.subheader(select2)
    col1, col2, col3, col4 = st.columns([5, 0.5, 1, 4])
    with col1:
        st.write("Total number of reviews for this hotel")
    with col2:
        st.markdown(count)
    with col3:
        st.markdown("reviews")
    with col4:
        st.markdown("")

    # Extract_Mean_Rating_per_date for the selected hotel
    hotel_data['date'] = hotel_data['date'].apply(lambda x: datetime.fromisoformat(x + ' 00:00:00'))
    base2 = hotel_data.groupby('date').aggregate({"rating": [list, "count", "mean"]}).reset_index()
    base2["count_rating_per_date"] = base2["rating"]["count"]
    base2["mean_rating_per_date"] = base2["rating"]["mean"]
    base2.drop(["rating"], axis=1, inplace=True)
    base2.set_index('date', inplace=True)
    base2 = base2.sort_index()
    Ratings_monthly = base2.resample('M').sum()

    # Figure 1
    count_stars = []
    a = []
    for i in range(5):
        count_stars.append(
            [hotel_data.rating.unique()[i], (hotel_data['rating'] == hotel_data.rating.unique()[i]).sum()])
    sorted(count_stars, key=itemgetter(1), reverse=True)
    for i in range(len(hotel_data.rating.unique())):
        a.append(hotel_data.rating.unique()[i])

    plt.figure(figsize=(16, 8))
    fig1 = px.pie(hotel_data, values=hotel_data["rating"].value_counts(), names=a)
    st.plotly_chart(fig1)

    clusters_names = ["Breakfast and Room service", "Interior design and Room decoration", "Pool and Beach",
                      "Relaxing atmosphere", "Food", "Animation team", "Disagreeing with people reviews",
                      "Very positive reviews", "Entertainment and Activities", "Reaction to negative reviews"]

    count_clusters = []
    for i in range(len(hotel_data.cluster_assignment.unique())):
        count_clusters.append([hotel_data.cluster_assignment.unique()[i],
                               (hotel_data['cluster_assignment'] == hotel_data.cluster_assignment.unique()[i]).sum()])
    sorted(count_clusters, key=itemgetter(1), reverse=True)

    clusters = []
    for i, j in (count_clusters):
        a = (hotel_data['cluster_assignment'] == i).sum()
        b = (hotel_data[hotel_data['cluster_assignment'] == i]['rating'] == 5).sum()
        c = (hotel_data[hotel_data['cluster_assignment'] == i]['rating'] == 4).sum()
        if a != 0:
            clusters.append([i, (b + c) / a * 100])
        else:
            clusters.append([i, -1])
    print(clusters)

    d = []
    for i in range(len(hotel_data.cluster_assignment.unique())):
        d.append(clusters_names[count_clusters[i][0]])

    Title1 = '<p style="color:#000080; font-size: 30px; font-style:bold">Clustering percentage </p>'
    st.markdown(Title1, unsafe_allow_html=True)
    plt.figure(figsize=(16, 8))
    f = px.pie(hotel_data, values=hotel_data["cluster_assignment"].value_counts(), names=d)

    col1, col2, col3 = st.columns([2, 2, 2])
    with col1:
        st.plotly_chart(f)

    with col2:
        st.empty()
    with col3:
        col3.write("yess")

    # Figure 2
    # st.title("<h4> In this study, using Time Series Forecasting, we will use <strong>count_rating_per_date</strong>, which is proportional to the number of customers visiting the hotel, we will plot the variation of this number, and we will analyze the factors influencing this variation (seasonality, trend, noise...) : this study will inform us about the key moments during which we should optimize our marketing </h4>")

    Title2 = '<p style="color:#000080; font-size: 30px; font-style:bold">Count Rating Per Date </p>'
    st.markdown(Title2, unsafe_allow_html=True)
    st.line_chart(Ratings_monthly['count_rating_per_date'])

    st.caption("We can notice the two crises that took place.\n")
    # if (max (Ratings_monthly[Ratings_monthly['date'in ['2016-01-31','2018-01-31']]]['count_rating_per_date'])) < 0.2 * max (Ratings_monthly[Ratings_monthly['date'<'2016-01-31']]['count_rating_per_date']) :
    st.caption(
        "The first one during the years 2016-2018, related to the terrorism that influenced the tourism for the following years.\n")
    st.caption(
        "The second one is related to Covid 19, we can notice the fall in 2019, and a fall of the rates during the years of the pandemic. \n")
    st.caption(
        "However, the rates do not completely cancel each other out, this can be explained by the decrease in hotel prices, and the periods during which Tunisia was able to recover and open its borders to tourism.\n")

    st.subheader('\n')
    st.subheader('\n')
    Title3 = '<p style="color:#000080; font-size: 30px; font-style:bold">Variation of Ratings Per Date</p>'
    st.markdown(Title3, unsafe_allow_html=True)
    st.line_chart(Ratings_monthly['mean_rating_per_date'])

    # Figure 3: estimation (Time Series Forecasting)
    import sys, os
    import warnings
    warnings.simplefilter(action='ignore')
    import tensorflow as tf
    import keras
    from keras.models import Sequential
    from keras.layers import Dense, SimpleRNN, LSTM, Activation, Dropout
    import math
    from sklearn.preprocessing import MinMaxScaler
    from sklearn.metrics import mean_squared_error
    def get_n_last_months(df, series_name, n_months):
        return df[series_name][-(n_months):]

    def plot_n_last_days(df, series_name, n_months):
        plt.figure(figsize=(10, 5))
        plt.plot(get_n_last_months(df, series_name, n_months), 'k-')
        plt.title('{0} number of reviews Time Series - {1} months'
                  .format(series_name, n_months))
        plt.xlabel('Recorded Month')
        plt.ylabel('Reading')
        plt.grid(alpha=0.3)

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

    def fit_LSTM(train_X, train_y, cell_units, epochs):
        model = Sequential()
        model.add(LSTM(cell_units, input_shape=(train_X.shape[1], 1)))
        model.add(Dense(1))
        model.compile(loss='mean_squared_error', optimizer='adam')
        model.fit(train_X, train_y, epochs=epochs, batch_size=64, verbose=0)
        return model

    def predict(X_init, n_steps, model):
        X_init = X_init.copy().reshape(1, -1, 1)
        preds = []
        for _ in range(n_steps):
            pred = model.predict(X_init)
            pred = pred[0][0]
            preds.append(pred)
            X_init[:, :-1, :] = X_init[:, 1:, :]  # replace first 11 values with 2nd through 12th
            X_init[:, -1, :] = pred  # replace 12th value with prediction
        return preds

    def predict_and_plot(X_init, y, model, title):
        y_preds = predict(test_X_init, n_steps=len(y), model=model)  # predict through length of y
        # Below ranges are to set x-axes
        start_range = range(1,
                            test_X_init.shape[0] + 1)  # starting at one through to length of test_X_init to plot X_init
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

    series_months = 194
    input_months = 12
    test_months = 48
    train_X, test_X_init, train_y, test_y = \
        (get_train_test_data(Ratings_monthly, 'count_rating_per_date', series_months,
                             input_months, test_months))
    model = fit_LSTM(train_X, train_y, cell_units=70, epochs=3000)
    predict_and_plot(test_X_init, test_y, model,
                     'Reviews counts Series: Test Data and LSTM Predictions')

    df1 = hotels.loc[hotels['address'] == 'Hammamet_Nabeul_Governorate', :]
    count_stars = []
    for i in range(5):
        count_stars.append([df1.rating.unique()[i], (df1['rating'] == df1.rating.unique()[i]).sum()])
    sorted(count_stars, key=itemgetter(1), reverse=True)
    print(count_stars)

    count_clusters = []
    for i in range(len(df1.cluster_assignment.unique())):
        count_clusters.append([df1.cluster_assignment.unique()[i],
                               (df1['cluster_assignment'] == df1.cluster_assignment.unique()[i]).sum()])
    sorted(count_clusters, key=itemgetter(1), reverse=True)
    print(count_clusters)

    clusters = []
    for i, j in (count_clusters):
        a = (df1['cluster_assignment'] == i).sum()
        b = (df1[df1['cluster_assignment'] == i]['rating'] == 5).sum()
        c = (df1[df1['cluster_assignment'] == i]['rating'] == 4).sum()
        if a != 0:
            clusters.append([i, (b + c) / a * 100])
        else:
            clusters.append([i, -1])
    print(clusters)

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
    col1, col2, col3, col4, col5 = st.columns([1, 2.5, 2.5, 2.5, 1])
    col1.write("")
    col2.markdown("<h3 style='text-align: center; color: #B56576;'>Hotels And Tourism</h3>", unsafe_allow_html=True)
    exp = col2.expander('show more')
    exp.markdown('<p class="big-font"> description here </p>', unsafe_allow_html=True)

    col3.markdown("<h3 style='text-align: center; color: #B56576;'>Food Retail Stores</h3>", unsafe_allow_html=True)
    expf = col3.expander('show more')
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
    expb = col4.expander('show more')
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

def food(st, **data):
    st.title('Try it yourself! Predicting Marketing Campaign Results')
    st.sidebar.header('Input Parameters')

    def user_input_features():
        Age = st.sidebar.slider('age', 0, 100, 30)
        Education = st.sidebar.selectbox(
            'Education:',
            ('Graduation', 'PhD', 'Master', 'Basic', '2n Cycle'))
        Marital_Status = st.sidebar.selectbox('Marital Status:', ('Married', 'Together', 'Alone', 'Widow'))
        Income = st.sidebar.number_input('Income:')
        Kidhome = st.sidebar.slider('Number of kids home', 0, 10, 1)
        Teenhome = st.sidebar.slider('Number of teens home', 0, 10, 1)
        Dt_Customer = st.sidebar.date_input('When did they become a registered customer')
        Recency = st.sidebar.number_input('Recency:', 0, 100, step=1)
        MntDairy = st.sidebar.number_input('Amount of Dairy purchased', 0, 1200, step=1)
        MntFruits = st.sidebar.number_input('Amount of fruits purchased:', 0, 200, step=1)
        MntMeatProducts = st.sidebar.number_input('Amount of meat products purchased:', 0, 2000, step=1)
        MntFishProducts = st.sidebar.number_input('Amount of fish products purchased:', 0, 2000, step=1)
        MntSweetProducts = st.sidebar.number_input('Amount of sweet products purchased:', 0, 2000, step=1)
        MntGold = st.sidebar.number_input('Amount of luxury products purchased:', 0, 2000, step=1)
        NumDealsPurchases = st.sidebar.number_input('Number of deals purchases', step=1)
        NumWebPurchases = st.sidebar.number_input('Number of web purchases:', step=1)
        NumCatalogPurchases = st.sidebar.number_input('Number of catalog purchases', step=1)
        NumStorePurchases = st.sidebar.number_input('Number of store purchases', step=1)
        NumWebVisitsMonth = st.sidebar.number_input('Number of web visits per month', step=1)
        AcceptedCmp1 = st.sidebar.slider('Accepted campaign 1:', 0, 1, 0)
        AcceptedCmp2 = st.sidebar.slider('Accepted campaign 2:', 0, 1, 0)
        AcceptedCmp3 = st.sidebar.slider('Accepted campaign 3:', 0, 1, 0)
        AcceptedCmp4 = st.sidebar.slider('Accepted campaign 4:', 0, 1, 0)
        AcceptedCmp5 = st.sidebar.slider('Accepted campaign 5:', 0, 1, 0)

        data = {'Age': Age,
                'Education': Education,
                'Marital_Status': Marital_Status,
                'Income': Income,
                'Kidhome': Kidhome,
                'Teenhome': Teenhome,
                'Dt_Customer': Dt_Customer,
                'Year_Customer': Dt_Customer.year,
                'Recency': Recency,
                'MntDairy': MntDairy,
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
                'AcceptedCmp3': AcceptedCmp3 == True,
                'AcceptedCmp4': AcceptedCmp4 == True,
                'AcceptedCmp5': AcceptedCmp5 == True,
                'AcceptedCmp1': AcceptedCmp1 == True,
                'AcceptedCmp2': AcceptedCmp2 == True,
                }
        features = pd.DataFrame(data, index=[0])
        return features

    data = user_input_features()
    data.insert(7, 'NumChildrenhome', data["Kidhome"] + data["Teenhome"])
    data.insert(15, 'Spendings',
                data["MntDairy"] + data["MntFruits"] + data["MntMeatProducts"] + data["MntFishProducts"] + data[
                    "MntSweetProducts"])
    data.insert(8, 'Haskids', data["NumChildrenhome"] > 0)
    data.insert(3, 'Years_Education', np.zeros)
    data["Years_Education"] = data['Education'].replace(['Basic', 'Graduation', 'PhD', 'Master', '2n Cycle'],
                                                        [6, 13, 21, 18, 9], inplace=False)
    data.drop('Education', axis=1, inplace=True)
    data["Income_status"] = np.nan
    lst = [data]

    for col in lst:
        col.loc[(col["Income"] >= 0) & (col["Income"] <= 9600), "Income_status"] = "low"
        col.loc[(col["Income"] > 9600) & (col["Income"] <= 36000), "Income_status"] = "middle"
        col.loc[col["Income"] > 36000, "Income_status"] = "high"

    st.subheader('User Input parameters')
    st.write(data)

    result = loaded_model.predict(data)
    resultproba = loaded_model.predict_proba(data)

    st.subheader('Class labels and their corresponding index number')
    st.write(['Refused Marketing Campaign', 'Accepted marketing Campaign'])

    st.subheader('Prediction results in probability')
    st.write(resultproba)
    print(result)

def sentiment(st, **data):
    st.title("Try it yourself! Sentiment Analysis with NLP")

    st.subheader("Client Feedback Sentiment Analysis")
    with st.form(key='nlpForm'):
        raw_text = st.text_area("Enter Text Here")
        submit_button = st.form_submit_button(label='Analyze')

    # layout
    col1, col2 = st.columns([1, 1])
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

def foodreport(st, **data):
    import streamlit as st
    from streamlit_echarts import st_echarts
    import pandas as pd
    import matplotlib.pyplot as plt
    import plotly.figure_factory as ff
    import squarify
    import plotly.graph_objects as go
    from plotly.offline import iplot
    import numpy as np
    from datetime import date
    import seaborn as sns
    import warnings
    from datetime import date
    from sklearn.preprocessing import LabelEncoder
    from sklearn.preprocessing import StandardScaler
    from sklearn import metrics
    import plotly.express as px
    import plotly.graph_objs as go

    warnings.filterwarnings("ignore")

    Titre_principal = '<h2 style="color:#000000; text-align:left; text-shadow: 0 0 1px #000000; font-size: 50px; font-style:bold">Food Retail Store: Marketing Campaign Report</h2>'
    st.markdown(Titre_principal, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([2.5, 0.5, 2])
    with col1:
        st.markdown("<h3>Get to know our Data:</h3>", unsafe_allow_html=True)
        data = pd.read_csv("tunisian_foodstore.csv")
        data.drop("Unnamed: 0", axis=1, inplace=True)
        st.write(data)

    with col2:
        st.write("")
    with col3:
        colors = ["#EFA48B", "#C6D8AF"]
        labels = "Failed Marketing campaign", "Marketing campaign successful"
        import plotly.express as px

        fig = px.pie(data.Response, values=data['Response'].value_counts().values, names=labels, color=labels,
                     color_discrete_map={'Thur': 'lightcyan',
                                         'Failed Marketing campaign': 'royalblue',
                                         'Marketing campaign successful': 'cyan',
                                         'Sun': 'darkblue'})
        fig.update_traces(hoverinfo='label+percent', textinfo='value')
        st.plotly_chart(fig)

    data.Marital_Status.value_counts()
    data.drop(data[data['Marital_Status'] == "Absurd"].index, inplace=True)
    data.drop(data[data['Marital_Status'] == "YOLO"].index, inplace=True)

    col1, col2, col3 = st.columns([2.5, 0.5, 2.5])
    with col1:
        # SQUARIFY TREE MAP
        x = 0
        y = 0
        width = 100
        height = 100

        education_level = data['Education'].value_counts().index
        values = data['Education'].value_counts().tolist()

        normed = squarify.normalize_sizes(values, width, height)
        rects = squarify.squarify(normed, x, y, width, height)

        colors = ['cyan', '#B8FFF9',
                  '#42C2FF', 'lightcyan',
                  '#51C4D3', '#132C33',
                  '#91C499', '#A89B9D',
                  '#91F5AD', '#97C8EB']

        shapes = []
        annotations = []
        counter = 0

        for r in rects:
            shapes.append(
                dict(
                    type='rect',
                    x0=r['x'],
                    y0=r['y'],
                    x1=r['x'] + r['dx'],
                    y1=r['y'] + r['dy'],
                    line=dict(width=2),
                    fillcolor=colors[counter]
                )
            )
            annotations.append(
                dict(
                    x=r['x'] + (r['dx'] / 2),
                    y=r['y'] + (r['dy'] / 2),
                    text=values[counter],
                    showarrow=False
                )
            )
            counter = counter + 1
            if counter >= len(colors):
                counter = 0

        # For hover text
        trace0 = go.Scatter(
            x=[r['x'] + (r['dx'] / 2) for r in rects],
            y=[r['y'] + (r['dy'] / 2) for r in rects],
            text=[str(v) for v in education_level],
            mode='text',
        )

        layout = dict(
            title='Number of Occupations <br> <i>(From our Sample Population)</i>',
            height=700,
            width=700,
            xaxis=dict(showgrid=False, zeroline=False),
            yaxis=dict(showgrid=False, zeroline=False),
            shapes=shapes,
            annotations=annotations,
            hovermode='closest'
        )

        # With hovertext
        figure = dict(data=[trace0], layout=layout)

        iplot(figure, filename='squarify-treemap')
        st.plotly_chart(figure, use_container_width=True)

    with col2:
        st.write("")
    with col3:
        st.write(" ")
        st.write(" ")
        st.markdown('<h4> Relationship Between Spendings and Income: </h4>', unsafe_allow_html=True)
        st.write(" ")
        st.write(' ')
        fig = px.scatter(data, x="Spendings", y="Income", trendline="ols", trendline_color_override="cyan",
                         color_discrete_sequence=['navy'])
        fig.update_layout({'plot_bgcolor': '#B1D0E0', 'paper_bgcolor': 'rgba(254,248,239,1)'})
        st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns([2.5, 0.5, 2.5])
    with col1:
        st.markdown('<h4> Marital Status in Relation with Income and Client Response:</h4>', unsafe_allow_html=True)

        fig = px.histogram(data, x="Marital_Status", y="Income",
                           color='Response', barmode='group',
                           height=400, color_discrete_sequence=['navy', 'cyan'])
        fig.update_layout({'plot_bgcolor': '#B1D0E0', 'paper_bgcolor': 'rgba(254,248,239,1)'})
        st.plotly_chart(fig, use_container_width=True)
        sns.set(rc={'axes.facecolor': '#B1D0E0', 'figure.facecolor': '#fef8ef'})

        fig = sns.catplot(x='NumWebVisitsMonth', y='Spendings', kind="swarm", data=data)
        st.pyplot(fig, use_container_width=True)
    with col2:
        st.write("")
    with col3:
        st.markdown('<h4> Sum of Income by Years of Education: </h4>', unsafe_allow_html=True)

        fig = px.histogram(data, x="Years_Education", y="Income",
                           color='Response', barmode='group',
                           height=400, color_discrete_sequence=['navy', 'cyan'])
        fig.update_layout({'plot_bgcolor': '#B1D0E0', 'paper_bgcolor': 'rgba(254,248,239,1)'})
        st.plotly_chart(fig, use_container_width=True)
        sns.set(rc={'axes.facecolor': '#B1D0E0', 'figure.facecolor': '#fef8ef'})
        fig = sns.catplot(x='NumDealsPurchases', y='Spendings', kind="swarm", data=data)
        st.pyplot(fig, use_container_width=True)

    # Creating Clusters:

    data = data.drop(["ID", "Education"], axis=1)
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
    dates = []
    for i in data['Dt_Customer']:
        i = i.date()
        dates.append(i)
    data['NumAllPurchases'] = data['NumWebPurchases'] + data['NumCatalogPurchases'] + data['NumStorePurchases']
    data['AverageCheck'] = round((data['Spendings'] / data['NumAllPurchases']), 1)
    data['ShareDealsPurchases'] = round((data['NumDealsPurchases'] / data['NumAllPurchases']) * 100, 1)
    data['TotalAcceptedCmp'] = data['AcceptedCmp1'] + data['AcceptedCmp2'] + data['AcceptedCmp3'] + data[
        'AcceptedCmp4'] + \
                               data['AcceptedCmp5'] + data['Response']

    data['Collected'] = date.today()
    data['Collected'] = pd.to_datetime(data['Collected'])
    data['Days_is_client'] = (data['Collected'] - data['Dt_Customer']).dt.days
    data.drop('Collected', axis=1, inplace=True)
    data["ActiveDays"] = data["Days_is_client"] - data['Recency']
    data['AverageCheck'] = np.where(data['AverageCheck'] > 200, 200, data['AverageCheck'])
    data_clustring = data[['AverageCheck', 'Days_is_client', 'NumAllPurchases']].copy()
    for i in data_clustring.columns:
        data_clustring[i] = StandardScaler().fit_transform(np.array(data_clustring[[i]]))
    from sklearn.cluster import KMeans

    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=228)
        km.fit(data_clustring)
        wcss.append(km.inertia_)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=4, covariance_type='spherical', max_iter=3000, random_state=228).fit(
        data_clustring)
    labels = gmm.predict(data_clustring)

    data['Cluster'] = labels
    data_re_clust = {
        0: 'Ordinary client',
        1: 'Elite client',
        2: 'Good client',
        3: 'Potential good client'
    }
    data['Cluster'] = data['Cluster'].map(data_re_clust)

    st.markdown('<h2> Clustering Part 1: Creating Customer Segments:</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2.5, 0.5, 2.5])

    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        fig = px.pie(data['Cluster'].value_counts().reset_index(), values='Cluster', names='index', width=700,
                     height=700)
        fig.update_traces(textposition='inside',
                          textinfo='percent + label',
                          hole=0.8,
                          marker=dict(colors=['#dd4124', '#009473', '#336b87', '#b4b4b4'],
                                      line=dict(color='white', width=2)),
                          hovertemplate='Clients: %{value}')

        fig.update_layout(annotations=[dict(text='Number of clients <br>by cluster',
                                            x=0.5, y=0.5, font_size=28, showarrow=False,
                                            font_family='monospace',
                                            font_color='black')],
                          showlegend=False)

        st.plotly_chart(fig)
    with col2:
        st.write('')
    with col3:
        plot = go.Figure()

        colors = ['#b4b4b4', '#dd4124', '#009473', '#336b87']
        names = ['Ordinary client', 'Elite client', 'Good client', 'Potential good client']

        for i in range(4):
            cl = names[i]
            plot.add_trace(go.Scatter3d(x=data.query("Cluster == @cl")['NumAllPurchases'],
                                        y=data.query("Cluster == @cl")['AverageCheck'],
                                        z=data.query("Cluster == @cl")['Days_is_client'],
                                        mode='markers',
                                        name=names[i],
                                        marker=dict(
                                            size=4,
                                            color=colors[i],
                                            opacity=0.6)))

        plot.update_traces(hovertemplate='Purchases: %{x} <br>Average Check: %{y} <br>Days is client: %{z}')

        plot.update_layout(width=800, height=800, autosize=True, showlegend=False,
                           scene=dict(xaxis=dict(title='Count of purchases', titlefont_color='black'),
                                      yaxis=dict(title='Average check', titlefont_color='black'),
                                      zaxis=dict(title='Days is client', titlefont_color='black')),
                           font=dict(family="monospace", color='black', size=12))

        st.plotly_chart(plot)

    st.markdown(
        "<h3> As we can see, we have four main clusters:</h3> <br> <h4> 1- Elite clients: Clients that make large "
        "purchases that cost a good amount of money (Average check and count of purchases are both on the higher "
        "end). </h4> <br> <h4> 2- Good clients: Clients who are in the average for both the check amount and the "
        "count of purchases. </h4> <br> <h4> 3- Ordinary clients: New clients, count purchase and average check "
        "amounts are very low compared to other clients.</h4><br><h4> 4- Potentially good clients: Old clients, "
        "amount purchased on the lower end but still higher than ordinary clients, same with count "
        "purchase.</h4><br>", unsafe_allow_html=True)

    data.drop(['Kidhome', 'Teenhome', 'Dt_Customer'], axis=1, inplace=True)
    data['Income'] = np.where(data['Income'] > 120000, 120000, data['Income'])

    import matplotlib.lines as lines

    data = data.rename(columns={'MntDairy': 'Dairy',
                                'MntFruits': 'Fruits',
                                'MntMeatProducts': 'Meat',
                                'MntFishProducts': 'Fish',
                                'MntSweetProducts': 'Sweet',
                                'MntGold': 'Gold'})

    cl = ['Ordinary client', 'Potential good client', 'Good client', 'Elite client']
    colors = {
        'Ordinary client': '#b4b4b4',
        'Potential good client': '#336b87',
        'Good client': '#009473',
        'Elite client': '#dd4124'
    }

    exp1 = st.expander('expand for more details:')
    with exp1:
        col1, col2, col3 = st.columns([3, 0.1, 3])
        with col1:

            fig = plt.figure(figsize=(13, 15))
            p = 1
            for i in range(len(data.columns.tolist()[4:10])):
                for k in cl:
                    plt.subplot(6, 4, p)
                    sns.set_style("white")
                    a = sns.kdeplot(data.query("Cluster == @k")[data.columns.tolist()[4:10][i]], color=colors[k],
                                    alpha=1,
                                    shade=True, linewidth=1.3, edgecolor='black')
                    plt.ylabel('')
                    plt.xlabel('')
                    plt.xticks(fontname='monospace')
                    plt.yticks([])
                    for j in ['right', 'left', 'top']:
                        a.spines[j].set_visible(False)
                        a.spines['bottom'].set_linewidth(1.2)
                    p += 1

            plt.figtext(0., 1.11, 'Distribution of purchases by clusters and product categories', fontname='monospace',
                        size=28.5,
                        color='black')

            plt.figtext(0.035, 1.03, 'Ordinary', fontname='monospace', size=20, color='#b4b4b4')
            plt.figtext(0.28, 1.03, 'Potential good', fontname='monospace', size=20, color='#336b87')
            plt.figtext(0.59, 1.03, 'Good', fontname='monospace', size=20, color='#009473')
            plt.figtext(0.83, 1.03, 'Elite', fontname='monospace', size=20, color='#dd4124')

            plt.figtext(1.015, 0.98, 'Dairy', fontname='monospace', size=20)
            plt.figtext(1.01, 0.814, 'Fruits', fontname='monospace', size=20)
            plt.figtext(1.02, 0.648, 'Meat', fontname='monospace', size=20)
            plt.figtext(1.02, 0.482, 'Fish', fontname='monospace', size=20)
            plt.figtext(1.012, 0.316, 'Sweet', fontname='monospace', size=20)
            plt.figtext(1.02, 0.15, 'Gold', fontname='monospace', size=20)

            l1 = lines.Line2D([0.99, 0.99], [1.08, 0], transform=fig.transFigure, figure=fig, color='black',
                              linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l1])
            l2 = lines.Line2D([0.0, 1.1], [1, 1], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l2])
            l3 = lines.Line2D([0.991, 1.1], [1, 1.08], transform=fig.transFigure, figure=fig, color='black',
                              linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l3])
            l4 = lines.Line2D([0, 1.1], [1.08, 1.08], transform=fig.transFigure, figure=fig, color='black',
                              linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l4])
            l5 = lines.Line2D([1.1, 1.1], [0, 1.08], transform=fig.transFigure, figure=fig, color='black',
                              linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l5])
            l6 = lines.Line2D([0, 0], [0, 1.08], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l6])
            l7 = lines.Line2D([0, 1.1], [0, 0], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l7])
            l8 = lines.Line2D([0, 1.1], [0.84, 0.84], transform=fig.transFigure, figure=fig, color='black',
                              linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l8])
            l9 = lines.Line2D([0, 1.1], [0.674, 0.674], transform=fig.transFigure, figure=fig, color='black',
                              linestyle='-',
                              linewidth=1.2)
            fig.lines.extend([l9])
            l10 = lines.Line2D([0, 1.1], [0.508, 0.508], transform=fig.transFigure, figure=fig, color='black',
                               linestyle='-',
                               linewidth=1.2)
            fig.lines.extend([l10])
            l11 = lines.Line2D([0, 1.1], [0.342, 0.342], transform=fig.transFigure, figure=fig, color='black',
                               linestyle='-',
                               linewidth=1.2)
            fig.lines.extend([l11])
            l12 = lines.Line2D([0, 1.1], [0.176, 0.176], transform=fig.transFigure, figure=fig, color='black',
                               linestyle='-',
                               linewidth=1.2)
            fig.lines.extend([l12])
            l13 = lines.Line2D([0.25, 0.25], [0, 1.08], transform=fig.transFigure, figure=fig, color='black',
                               linestyle='-',
                               linewidth=1.2)
            fig.lines.extend([l13])
            l14 = lines.Line2D([0.495, 0.495], [0, 1.08], transform=fig.transFigure, figure=fig, color='black',
                               linestyle='-',
                               linewidth=1.2)
            fig.lines.extend([l14])
            l15 = lines.Line2D([0.745, 0.745], [0, 1.08], transform=fig.transFigure, figure=fig, color='black',
                               linestyle='-',
                               linewidth=1.2)
            fig.lines.extend([l15])

            plt.figtext(1.027, 1.02, '''Customers
            clusters''', fontname='monospace', size=12, rotation=41, ha='center')
            plt.figtext(1.025, 1.003, '''Products
            category''', fontname='monospace', size=12, rotation=41)

            y = 0.94
            for i in range(6):
                plt.figtext(0.998, y, 'mean values:', fontname='monospace', size=13)
                y -= 0.1666

            y = 0.92
            for i in data.columns.tolist()[4:10]:
                plt.figtext(1.027, y, round(data.query("Cluster == 'Ordinary client'")[i].mean(), 1),
                            fontname='monospace',
                            size=14,
                            color='#b4b4b4')
                y -= 0.1666

            y = 0.9
            for i in data.columns.tolist()[4:10]:
                plt.figtext(1.027, y, round(data.query("Cluster == 'Potential good client'")[i].mean(), 1),
                            fontname='monospace',
                            size=14, color='#336b87')
                y -= 0.1666

            y = 0.88
            for i in data.columns.tolist()[4:10]:
                plt.figtext(1.027, y, round(data.query("Cluster == 'Good client'")[i].mean(), 1), fontname='monospace',
                            size=14,
                            color='#009473')
                y -= 0.1666

            y = 0.86
            for i in data.columns.tolist()[4:10]:
                plt.figtext(1.027, y, round(data.query("Cluster == 'Elite client'")[i].mean(), 1), fontname='monospace',
                            size=14,
                            color='#dd4124')
                y -= 0.1666

            fig.tight_layout(h_pad=2)
            fig.set_facecolor('#0000')

            st.pyplot(fig)
            st.markdown('<br>', unsafe_allow_html=True)
            fig = px.scatter(data, x="Income", y="Spendings", color="Cluster")
            fig.update_layout({'plot_bgcolor': 'rgba(254,248,239,1)', 'paper_bgcolor': 'rgba(254,248,239,1)'})
            st.plotly_chart(fig)
        with col2:
            st.write('')
        with col3:
            fig = plt.figure(figsize=(15, 12))
            k = 1

            for i in cl:
                ass = data.groupby(['Cluster']).agg(
                    {'Dairy': 'sum', 'Fruits': 'sum', 'Meat': 'sum', 'Fish': 'sum', 'Sweet': 'sum',
                     'Gold': 'sum'}).transpose().reset_index().rename(
                    columns={'index': 'Category'})[['Category', i]]
                plt.subplot(2, 2, k)
                plt.title(i, size=20, x=0.5, y=1.03)
                a = sns.barplot(data=ass, x='Category', y=i, color=colors[i],
                                linestyle="-", linewidth=1,
                                edgecolor="black")
                plt.xticks(fontname='monospace', size=13, color='black')
                plt.yticks(fontname='monospace', size=13, color='black')
                plt.xlabel('')
                plt.ylabel('')
                for p in a.patches:
                    height = p.get_height()
                    a.annotate(f'{round((height / sum(ass[i])) * 100, 1)}%',
                               (p.get_x() + p.get_width() / 2, p.get_height()),
                               ha='center', va='center',
                               size=11,
                               xytext=(0, 5),
                               textcoords='offset points',
                               fontname='monospace', color='black')

                for j in ['right', 'top']:
                    a.spines[j].set_visible(False)
                for j in ['bottom', 'left']:
                    a.spines[j].set_linewidth(1.5)
                k += 1

            plt.figtext(0.2, 1.05, 'What do customers from different clusters buy?', fontname='monospace', size=25)
            fig.tight_layout(h_pad=3)
            fig.set_facecolor('#0000')
            st.pyplot(fig)
            fig = plt.figure(figsize=(15, 10))
            st.markdown("<br>", unsafe_allow_html=True)
            plt.title('Participation of customer clusters in marketing campaigns', size=25, x=0.5, y=1.1)
            plt.grid(color='gray', linestyle=':', axis='y', alpha=0.8, zorder=0, dashes=(1, 7))
            a = sns.barplot(x='Cmp', y='value', hue='Cluster',
                            data=data.groupby(['Cluster']).agg({'AcceptedCmp1': 'sum', 'AcceptedCmp2': 'sum',
                                                                'AcceptedCmp3': 'sum', 'AcceptedCmp4': 'sum',
                                                                'AcceptedCmp5': 'sum',
                                                                'Response': 'sum'}).stack().reset_index().rename(
                                columns={'level_1': 'Cmp', 0: 'value'}),
                            dodge=False, palette=['#dd4124', '#009473', '#b4b4b4', '#336b87'])
            plt.xticks(fontname='monospace', size=16, color='black')
            plt.yticks(fontname='monospace', size=16, color='black')
            plt.xlabel('')
            plt.ylabel('')
            for j in ['right', 'top']:
                a.spines[j].set_visible(False)
            for j in ['bottom', 'left']:
                a.spines[j].set_linewidth(1.5)
            fig.set_facecolor('#0000')
            st.pyplot(fig)

    data_clustring = data[['AverageCheck', 'NumWebPurchases', 'NumWebVisitsMonth']].copy()
    for i in data_clustring.columns:
        data_clustring[i] = StandardScaler().fit_transform(np.array(data_clustring[[i]]))

    from sklearn.cluster import KMeans

    wcss = []
    for i in range(1, 11):
        km = KMeans(n_clusters=i, random_state=228)
        km.fit(data_clustring)
        wcss.append(km.inertia_)

    from sklearn.mixture import GaussianMixture

    gmm = GaussianMixture(n_components=3, covariance_type='spherical', max_iter=3000, random_state=228).fit(
        data_clustring)
    labels = gmm.predict(data_clustring)

    data['Cluster'] = labels
    data_re_clust = {
        0: 'Mixed',
        1: 'Digital Marketing',
        2: 'Direct Marketing',
    }
    data['Cluster'] = data['Cluster'].map(data_re_clust)

    import plotly.graph_objs as go
    st.markdown('<h2> Clustering Part 2: Creating Marketing Channels Segments:</h2>', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([2.5, 0.5, 2.5])

    with col1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        import plotly.express as px
        fig = px.pie(data['Cluster'].value_counts().reset_index(), values='Cluster', names='index', width=700,
                     height=700)
        fig.update_traces(textposition='inside',
                          textinfo='percent + label',
                          hole=0.8,
                          marker=dict(colors=['#dd4124', '#009473', '#336b87', '#b4b4b4'],
                                      line=dict(color='white', width=2)),
                          hovertemplate='Clients: %{value}')

        fig.update_layout(annotations=[dict(text='Number of clients <br>by cluster',
                                            x=0.5, y=0.5, font_size=28, showarrow=False,
                                            font_family='monospace',
                                            font_color='black')],
                          showlegend=False)

        st.plotly_chart(fig)
    with col2:
        st.write('')
    with col3:
        plot = go.Figure()

        colors = ['#dd4124', 'navy', '#009473']
        names = ['Mixed', 'Digital Marketing', 'Direct Marketing']

        for i in range(3):
            cl = names[i]
            plot.add_trace(go.Scatter3d(x=data.query("Cluster == @cl")['NumWebVisitsMonth'],
                                        y=data.query("Cluster == @cl")['NumWebPurchases'],
                                        z=data.query("Cluster == @cl")['AverageCheck'],
                                        mode='markers',
                                        name=names[i],
                                        marker=dict(
                                            size=2,
                                            color=colors[i],
                                            opacity=0.6)))

        plot.update_traces(
            hovertemplate='Monthly web vists: %{x} <br>Number of purchases: %{y} <br>Average Check: %{z}')

        plot.update_layout(width=800, height=800, autosize=True, showlegend=False,
                           scene=dict(xaxis=dict(title='Number of web visits per month', titlefont_color='black'),
                                      yaxis=dict(title='Number of web purchases', titlefont_color='black'),
                                      zaxis=dict(title='Average check', titlefont_color='black')),
                           font=dict(family="monospace", color='black', size=12))

        st.plotly_chart(plot)

    st.markdown(
        "<h3> We have three main clusters here:</h3> <br> <h4> Direct Marketing: Number of web visits per month "
        "is very low / average for the most part except a few clients, average check is mainly average and number "
        "of web purchases is fairly low. "
        "</h4> <br> <h4> Digtal Marketing: Both the Number of web visits per month and web purchases are on the "
        "higher end, average check is lower than that of clients that recieved direct marketing. "
        "</h4> <br> <h4>Mixed: Number of web visits per month is on the mid / higher end, number of web purchases "
        "is very low and the Average check is very low. "
        "</h4><br>", unsafe_allow_html=True)

    exp2 = st.expander('expand for more details:')
    with exp2:
        col1, col2, col3 = st.columns([3, 0.1, 3])
        with col1:
            fig = plt.figure(figsize=(15, 10))
            plt.title('Participation of customer clusters in marketing campaigns', size=25, x=0.5, y=1.1)
            plt.grid(color='gray', linestyle=':', axis='y', alpha=0.8, zorder=0, dashes=(1, 7))
            a = sns.barplot(x='Cmp', y='value', hue='Cluster',
                            data=data.groupby(['Cluster']).agg({'AcceptedCmp1': 'sum', 'AcceptedCmp2': 'sum',
                                                                'AcceptedCmp3': 'sum', 'AcceptedCmp4': 'sum',
                                                                'AcceptedCmp5': 'sum',
                                                                'Response': 'sum'}).stack().reset_index().rename(
                                columns={'level_1': 'Cmp', 0: 'value'}),
                            dodge=False, palette=['#dd4124', '#009473', '#b4b4b4', '#336b87'])
            plt.xticks(fontname='monospace', size=16, color='black')
            plt.yticks(fontname='monospace', size=16, color='black')
            plt.xlabel('')
            plt.ylabel('')
            for j in ['right', 'top']:
                a.spines[j].set_visible(False)
            for j in ['bottom', 'left']:
                a.spines[j].set_linewidth(1.5)
            fig.set_facecolor('#0000')
            st.pyplot(fig)
        with col2:
            st.write('')
        with col3:
            fig = plt.figure(figsize=(15, 10))
            plt.title('Participation of customer clusters in marketing campaigns', size=25, x=0.5, y=1.1)
            plt.grid(color='gray', linestyle=':', axis='y', alpha=0.8, zorder=0, dashes=(1, 7))
            a = sns.barplot(x='Cmp', y='value', hue='Cluster',
                            data=data.groupby(['Cluster']).agg({'NumWebPurchases': 'sum', 'NumDealsPurchases': 'sum',
                                                                'NumStorePurchases': 'sum',
                                                                'NumAllPurchases': 'sum'}).stack().reset_index().rename(
                                columns={'level_1': 'Cmp', 0: 'value'}),
                            dodge=False, palette=['#dd4124', '#009473', '#b4b4b4', '#336b87'])
            plt.xticks(fontname='monospace', size=16, color='black')
            plt.yticks(fontname='monospace', size=16, color='black')
            plt.xlabel('')
            plt.ylabel('')
            for j in ['right', 'top']:
                a.spines[j].set_visible(False)
            for j in ['bottom', 'left']:
                a.spines[j].set_linewidth(1.5)
            fig.set_facecolor('#0000')
            st.pyplot(fig)


app = MultiPage()
app.st = st

app.add_app("Home Page", home, initial_page=True)
app.add_app("1", home)
app.add_app("2", foodreport)
app.add_app("3",food)
app.add_app("4", hotel)
app.add_app("5", hammamet)
app.add_app("6", sentiment)

app.run()
