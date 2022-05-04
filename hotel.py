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
#new_title = '<p style="color:Blue; font-size: 28px;">Please select an hotel from the list to get **an overview of the customer feedbacks**, **the variation of these feedbacks over time**, as well as **the variation of the number of customers visiting this hotel over time**.</p>'
#st.markdown(new_title, unsafe_allow_html=True)
st.markdown("Please select an hotel from the list to get **an overview of the customer feedbacks**, **the variation of these feedbacks over time**, as well as **the variation of the number of customers visiting this hotel over time**.")

DATA_URL="./hotel_clusters.csv"
@st.cache(persist=True)  #( If you have a different use case where the data does not change so very often, you can simply use this)
def load_data():
    data=pd.read_csv(DATA_URL)
    return data
hotels=load_data()



select = st.sidebar.selectbox('Select an Address',hotels['address'].unique())
#get the address selected in the selectbox
address_data = hotels[hotels['address'] == select ]
print(address_data['hotel'].value_counts())

select2 = st.sidebar.selectbox('Select an Hotel',address_data['hotel'].unique())
#select_cluster = st.sidebar.radio("Topic", ('Cluster0', 'Cluster1', 'Cluster2', 'Cluster3', 'Cluster4', 'Cluster5', 'Cluster6', 'Cluster7', 'Cluster8', 'Cluster9'))

hotel_data=address_data.loc[address_data['hotel']==select2,:]
hotel_data.drop(columns=['count_rating_per_date','tf_idf',	'distances', 'mean_rating_per_date', 'count_rating_per_year', 'mean_rating_per_year', 'avg_rating'],inplace=True)
count=(address_data['hotel']==select2).sum()

st.subheader('\n')
st.subheader(select2)
col1, col2, col3, col4 = st.columns([5,0.5,1,4])
with col1:
    st.write("Total number of reviews for this hotel")
with col2:
    st.markdown(count)
with col3:
    st.markdown("reviews")
with col4:
    st.markdown("")






#Extract_Mean_Rating_per_date for the selected hotel
hotel_data['date']=hotel_data['date'].apply(lambda x: datetime.fromisoformat(x+' 00:00:00'))
base2 = hotel_data.groupby('date').aggregate({"rating": [list, "count","mean"]}).reset_index()
base2["count_rating_per_date"] = base2["rating"]["count"]
base2["mean_rating_per_date"] = base2["rating"]["mean"]
base2.drop(["rating"], axis=1, inplace=True)
base2.set_index('date', inplace=True)
base2=base2.sort_index()
Ratings_monthly = base2.resample('M').sum()



#Figure 1
count_stars=[]
a=[]
for i in range (5):
  count_stars.append([hotel_data.rating.unique()[i],(hotel_data['rating']==hotel_data.rating.unique()[i]).sum()])
sorted(count_stars, key=itemgetter(1), reverse=True)
for i in range (len(hotel_data.rating.unique())):
  a.append(hotel_data.rating.unique()[i])

plt.figure(figsize=(16,8))
fig1=px.pie(hotel_data, values=hotel_data["rating"].value_counts(), names=a)
st.plotly_chart(fig1)


clusters_names=["Breakfast and Room service", "Interior design and Room decoration", "Pool and Beach", "Relaxing atmosphere", "Food", "Animation team", "Disagreeing with people reviews", "Very positive reviews", "Entertainment and Activities", "Reaction to negative reviews"]

count_clusters=[]
for i in range (len(hotel_data.cluster_assignment.unique())):
  count_clusters.append([hotel_data.cluster_assignment.unique()[i],(hotel_data['cluster_assignment']==hotel_data.cluster_assignment.unique()[i]).sum()])
sorted(count_clusters, key=itemgetter(1), reverse=True)


clusters=[]
for i,j in (count_clusters):
    a=(hotel_data['cluster_assignment']==i).sum()
    b=(hotel_data[hotel_data['cluster_assignment']==i]['rating']==5).sum()
    c=(hotel_data[hotel_data['cluster_assignment']==i]['rating']==4).sum()
    if a!=0 :
        clusters.append([i,(b+c)/a*100])
    else :
        clusters.append([i,-1])
print(clusters)

d=[]
for i in range (len(hotel_data.cluster_assignment.unique())):
    d.append(clusters_names[count_clusters[i][0]])



Title1 = '<p style="color:#000080; font-size: 30px; font-style:bold">Clustering percentage </p>'
st.markdown(Title1, unsafe_allow_html=True)
plt.figure(figsize=(16,8))
f=px.pie(hotel_data, values=hotel_data["cluster_assignment"].value_counts(), names=d)

col1, col2,col3= st.columns([2,2,2])
with col1:
    st.plotly_chart(f)
    
with col2:
    st.empty()
with col3:
    col3.write("yess")
    

#Figure 2
#st.title("<h4> In this study, using Time Series Forecasting, we will use <strong>count_rating_per_date</strong>, which is proportional to the number of customers visiting the hotel, we will plot the variation of this number, and we will analyze the factors influencing this variation (seasonality, trend, noise...) : this study will inform us about the key moments during which we should optimize our marketing </h4>")


Title2 = '<p style="color:#000080; font-size: 30px; font-style:bold">Count Rating Per Date </p>'
st.markdown(Title2, unsafe_allow_html=True)
st.line_chart(Ratings_monthly['count_rating_per_date'])

st.caption("We can notice the two crises that took place.\n")
#if (max (Ratings_monthly[Ratings_monthly['date'in ['2016-01-31','2018-01-31']]]['count_rating_per_date'])) < 0.2 * max (Ratings_monthly[Ratings_monthly['date'<'2016-01-31']]['count_rating_per_date']) :  
st.caption("The first one during the years 2016-2018, related to the terrorism that influenced the tourism for the following years.\n")
st.caption("The second one is related to Covid 19, we can notice the fall in 2019, and a fall of the rates during the years of the pandemic. \n")
st.caption("However, the rates do not completely cancel each other out, this can be explained by the decrease in hotel prices, and the periods during which Tunisia was able to recover and open its borders to tourism.\n")

st.subheader('\n')
st.subheader('\n')
Title3 = '<p style="color:#000080; font-size: 30px; font-style:bold">Variation of Ratings Per Date</p>'
st.markdown(Title3, unsafe_allow_html=True)
st.line_chart(Ratings_monthly['mean_rating_per_date'])





df1=hotels.loc[hotels['address']=='Hammamet_Nabeul_Governorate',:]
count_stars=[]
for i in range (5):
  count_stars.append([df1.rating.unique()[i],(df1['rating']==df1.rating.unique()[i]).sum()])
sorted(count_stars, key=itemgetter(1), reverse=True)
print(count_stars)

count_clusters=[]
for i in range (len(df1.cluster_assignment.unique())):
  count_clusters.append([df1.cluster_assignment.unique()[i],(df1['cluster_assignment']==df1.cluster_assignment.unique()[i]).sum()])
sorted(count_clusters, key=itemgetter(1), reverse=True)
print(count_clusters)

clusters=[]
for i,j in (count_clusters):
    a=(df1['cluster_assignment']==i).sum()
    b=(df1[df1['cluster_assignment']==i]['rating']==5).sum()
    c=(df1[df1['cluster_assignment']==i]['rating']==4).sum()
    if a!=0 :
        clusters.append([i,(b+c)/a*100])
    else :
        clusters.append([i,-1])
print(clusters)
