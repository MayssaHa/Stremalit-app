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
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
warnings.filterwarnings("ignore")
data=pd.read_csv("tunisian_foodstore.csv")
data.drop("Unnamed: 0",axis=1,inplace=True)
st.write(data)

sns.set(style="darkgrid")

colors = ["#EFA48B", "#C6D8AF"]
labels ="Failed Marketing campaign on client", "Marketing campaign successful on client"
figure=plt.figure(figsize=(16,8))
plt.suptitle('\n Marketing campaign Response', fontsize=20)
data["Response"].value_counts().plot.pie(explode=[0,0.25], autopct='%1.2f%%', shadow=True, colors=colors,
                                             labels=labels, fontsize=12, startangle=25, ylabel='')

st.pyplot(figure)

x = 0
y = 0
width = 100
height = 100

education_level = data['Education'].value_counts().index
values = data['Education'].value_counts().tolist()

normed = squarify.normalize_sizes(values, width, height)
rects = squarify.squarify(normed, x, y, width, height)

colors = ['#9AD1D4', '#ACF7C1',
          '#CECFC7', '#CBC5EA',
          '#45CB85', '#808F85',
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

import plotly.express as px

fig = px.scatter(data, x="Spendings", y="Income", trendline="ols",trendline_color_override="red")
st.plotly_chart(fig, use_container_width=True)

data.Marital_Status.value_counts()
data.drop(data[data['Marital_Status'] == "Absurd"].index, inplace = True)
data.drop(data[data['Marital_Status'] == "YOLO"].index, inplace = True)
data=data.drop(["ID","Education"],axis=1)

data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"])
dates=[]
for i in data['Dt_Customer']:
    i=i.date()
    dates.append(i)
data['NumAllPurchases'] = data['NumWebPurchases']+data['NumCatalogPurchases']+data['NumStorePurchases']
data['AverageCheck'] = round((data['Spendings'] / data['NumAllPurchases']), 1)
data['ShareDealsPurchases'] = round((data['NumDealsPurchases'] / data['NumAllPurchases']) * 100, 1)
data['TotalAcceptedCmp'] = data['AcceptedCmp1']+data['AcceptedCmp2']+data['AcceptedCmp3']+data['AcceptedCmp4']+data['AcceptedCmp5']+data['Response']
from datetime import date
data['Collected'] = date.today()
data['Collected'] = pd.to_datetime(data['Collected'])
data['Days_is_client'] = (data['Collected'] - data['Dt_Customer']).dt.days
data.drop('Collected',axis=1, inplace =True)
data["ActiveDays"]=data["Days_is_client"]-data['Recency']
data['AverageCheck'] = np.where(data['AverageCheck'] > 200, 200, data['AverageCheck'])
data_clustring=data[['AverageCheck', 'Days_is_client', 'NumAllPurchases']].copy()
for i in data_clustring.columns:
    data_clustring[i]=StandardScaler().fit_transform(np.array(data_clustring[[i]]))
from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=228)
    km.fit(data_clustring)
    wcss.append(km.inertia_)

fig1= plt.figure(figsize=(12, 8))
plt.title('The Elbow Method', size=25, y=1.03, fontname='monospace')
plt.grid(color='gray', linestyle=':', axis='y', alpha=0.8, zorder=0, dashes=(1, 7))
a = sns.lineplot(x=range(1, 11), y=wcss, color='#336b87', linewidth=3)
sns.scatterplot(x=range(1, 11), y=wcss, color='#336b87', s=60, edgecolor='black', zorder=5)
plt.ylabel('WCSS', size=14, fontname='monospace')
plt.xlabel('Number of clusters', size=14, fontname='monospace')
plt.xticks(size=12, fontname='monospace')
plt.yticks(size=12, fontname='monospace')

for j in ['right', 'top']:
    a.spines[j].set_visible(False)
a.spines['bottom'].set_linewidth(1.3)
a.spines['left'].set_linewidth(1.3)

plt.annotate('''Optimal number
of clusters''', xy=(4.05, 2400), xytext=(5, 2800),
             arrowprops=dict(facecolor='steelblue', arrowstyle="->", connectionstyle="arc3,rad = 0.4", color='#dd4124'),
             fontsize=13, fontfamily='monospace', ha='center', color='#dd4124')

st.pyplot(fig1)
from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 4, covariance_type = 'spherical', max_iter = 3000, random_state = 228).fit(data_clustring)
labels = gmm.predict(data_clustring)

data['Cluster'] = labels
data_re_clust = {
    0: 'Ordinary client',
    1: 'Elite client',
    2: 'Good client',
    3: 'Potential good client'
}
data['Cluster'] = data['Cluster'].map(data_re_clust)
import plotly.express as px

fig = px.pie(data['Cluster'].value_counts().reset_index(), values='Cluster', names='index', width=700, height=700)
fig.update_traces(textposition='inside',
                  textinfo='percent + label',
                  hole=0.8,
                  marker=dict(colors=['#dd4124', '#009473', '#336b87', '#b4b4b4'], line=dict(color='white', width=2)),
                  hovertemplate='Clients: %{value}')

fig.update_layout(annotations=[dict(text='Number of clients <br>by cluster',
                                    x=0.5, y=0.5, font_size=28, showarrow=False,
                                    font_family='monospace',
                                    font_color='black')],
                  showlegend=False)

st.plotly_chart(fig)

import plotly.graph_objs as go
plot = go.Figure()

colors = ['#b4b4b4', '#dd4124', '#009473', '#336b87']
names = ['Ordinary client', 'Elite client', 'Good client', 'Potential good client']

for i in range(4):
    cl = names[i]
    plot.add_trace(go.Scatter3d(x = data.query("Cluster == @cl")['NumAllPurchases'],
                                y = data.query("Cluster == @cl")['AverageCheck'],
                                z = data.query("Cluster == @cl")['Days_is_client'],
                                mode = 'markers',
                                name = names[i],
                                marker = dict(
                                    size = 4,
                                    color = colors[i],
                                    opacity = 0.6)))

plot.update_traces(hovertemplate = 'Purchases: %{x} <br>Average Check: %{y} <br>Days is client: %{z}')

plot.update_layout(width = 800, height = 800, autosize = True, showlegend = False,
                   scene = dict(xaxis = dict(title = 'Count of purchases', titlefont_color = 'black'),
                                yaxis = dict(title = 'Average check', titlefont_color = 'black'),
                                zaxis = dict(title = 'Days is client', titlefont_color = 'black')),
                   font = dict(family = "monospace", color  = 'black', size = 12),
                   title_text = 'Customers clusters', title_x = 0.5)

st.plotly_chart(plot)

data.drop(['Kidhome', 'Teenhome', 'Dt_Customer'], axis = 1, inplace = True)
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

fig = plt.figure(figsize=(13, 15))
p = 1
for i in range(len(data.columns.tolist()[4:10])):
    for k in cl:
        plt.subplot(6, 4, p)
        sns.set_style("white")
        a = sns.kdeplot(data.query("Cluster == @k")[data.columns.tolist()[4:10][i]], color=colors[k], alpha=1,
                        shade=True, linewidth=1.3, edgecolor='black')
        plt.ylabel('')
        plt.xlabel('')
        plt.xticks(fontname='monospace')
        plt.yticks([])
        for j in ['right', 'left', 'top']:
            a.spines[j].set_visible(False)
            a.spines['bottom'].set_linewidth(1.2)
        p += 1

plt.figtext(0., 1.11, 'Distribution of purchases by clusters and product categories', fontname='monospace', size=28.5,
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

l1 = lines.Line2D([0.99, 0.99], [1.08, 0], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                  linewidth=1.2)
fig.lines.extend([l1])
l2 = lines.Line2D([0.0, 1.1], [1, 1], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                  linewidth=1.2)
fig.lines.extend([l2])
l3 = lines.Line2D([0.991, 1.1], [1, 1.08], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                  linewidth=1.2)
fig.lines.extend([l3])
l4 = lines.Line2D([0, 1.1], [1.08, 1.08], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                  linewidth=1.2)
fig.lines.extend([l4])
l5 = lines.Line2D([1.1, 1.1], [0, 1.08], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                  linewidth=1.2)
fig.lines.extend([l5])
l6 = lines.Line2D([0, 0], [0, 1.08], transform=fig.transFigure, figure=fig, color='black', linestyle='-', linewidth=1.2)
fig.lines.extend([l6])
l7 = lines.Line2D([0, 1.1], [0, 0], transform=fig.transFigure, figure=fig, color='black', linestyle='-', linewidth=1.2)
fig.lines.extend([l7])
l8 = lines.Line2D([0, 1.1], [0.84, 0.84], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                  linewidth=1.2)
fig.lines.extend([l8])
l9 = lines.Line2D([0, 1.1], [0.674, 0.674], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                  linewidth=1.2)
fig.lines.extend([l9])
l10 = lines.Line2D([0, 1.1], [0.508, 0.508], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                   linewidth=1.2)
fig.lines.extend([l10])
l11 = lines.Line2D([0, 1.1], [0.342, 0.342], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                   linewidth=1.2)
fig.lines.extend([l11])
l12 = lines.Line2D([0, 1.1], [0.176, 0.176], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                   linewidth=1.2)
fig.lines.extend([l12])
l13 = lines.Line2D([0.25, 0.25], [0, 1.08], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                   linewidth=1.2)
fig.lines.extend([l13])
l14 = lines.Line2D([0.495, 0.495], [0, 1.08], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
                   linewidth=1.2)
fig.lines.extend([l14])
l15 = lines.Line2D([0.745, 0.745], [0, 1.08], transform=fig.transFigure, figure=fig, color='black', linestyle='-',
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
    plt.figtext(1.027, y, round(data.query("Cluster == 'Ordinary client'")[i].mean(), 1), fontname='monospace', size=14,
                color='#b4b4b4')
    y -= 0.1666

y = 0.9
for i in data.columns.tolist()[4:10]:
    plt.figtext(1.027, y, round(data.query("Cluster == 'Potential good client'")[i].mean(), 1), fontname='monospace',
                size=14, color='#336b87')
    y -= 0.1666

y = 0.88
for i in data.columns.tolist()[4:10]:
    plt.figtext(1.027, y, round(data.query("Cluster == 'Good client'")[i].mean(), 1), fontname='monospace', size=14,
                color='#009473')
    y -= 0.1666

y = 0.86
for i in data.columns.tolist()[4:10]:
    plt.figtext(1.027, y, round(data.query("Cluster == 'Elite client'")[i].mean(), 1), fontname='monospace', size=14,
                color='#dd4124')
    y -= 0.1666

fig.tight_layout(h_pad=2)
st.pyplot(fig)

fig = plt.figure(figsize=(15, 12))
k = 1

for i in cl:
    ass = data.groupby(['Cluster']).agg({'Dairy': 'sum', 'Fruits': 'sum', 'Meat': 'sum', 'Fish': 'sum', 'Sweet': 'sum',
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
        a.annotate(f'{round((height / sum(ass[i])) * 100, 1)}%', (p.get_x() + p.get_width() / 2, p.get_height()),
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
st.pyplot(fig)

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

st.pyplot(fig)

data_clustring=data[['AverageCheck', 'NumWebPurchases', 'NumWebVisitsMonth']].copy()
for i in data_clustring.columns:
    data_clustring[i]=StandardScaler().fit_transform(np.array(data_clustring[[i]]))

from sklearn.cluster import KMeans

wcss = []
for i in range(1, 11):
    km = KMeans(n_clusters=i, random_state=228)
    km.fit(data_clustring)
    wcss.append(km.inertia_)

fig=plt.figure(figsize=(12, 8))
plt.title('The Elbow Method', size=25, y=1.03, fontname='monospace')
plt.grid(color='gray', linestyle=':', axis='y', alpha=0.8, zorder=0, dashes=(1, 7))
a = sns.lineplot(x=range(1, 11), y=wcss, color='#336b87', linewidth=3)
sns.scatterplot(x=range(1, 11), y=wcss, color='#336b87', s=60, edgecolor='black', zorder=5)
plt.ylabel('WCSS', size=14, fontname='monospace')
plt.xlabel('Number of clusters', size=14, fontname='monospace')
plt.xticks(size=12, fontname='monospace')
plt.yticks(size=12, fontname='monospace')

for j in ['right', 'top']:
    a.spines[j].set_visible(False)
a.spines['bottom'].set_linewidth(1.3)
a.spines['left'].set_linewidth(1.3)

plt.annotate('''Optimal number
of clusters''', xy=(3, 3200), xytext=(5, 2800),
             arrowprops=dict(facecolor='steelblue', arrowstyle="->", connectionstyle="arc3,rad = 0.4", color='#dd4124'),
             fontsize=13, fontfamily='monospace', ha='center', color='#dd4124')

st.pyplot(fig)

from sklearn.mixture import GaussianMixture
gmm = GaussianMixture(n_components = 3, covariance_type = 'spherical', max_iter = 3000, random_state = 228).fit(data_clustring)
labels = gmm.predict(data_clustring)

data['Cluster'] = labels
data_re_clust = {
    0: 'Mixed',
    1: 'Digital Marketing',
    2: 'Direct Marketing',
}
data['Cluster'] = data['Cluster'].map(data_re_clust)

import plotly.graph_objs as go
plot = go.Figure()

colors = ['#b4b4b4', '#dd4124', '#009473']
names = ['Mixed', 'Digital Marketing','Direct Marketing']

for i in range(3):
    cl = names[i]
    plot.add_trace(go.Scatter3d(x = data.query("Cluster == @cl")['NumWebVisitsMonth'],
                                y = data.query("Cluster == @cl")['NumWebPurchases'],
                                z = data.query("Cluster == @cl")['AverageCheck'],
                                mode = 'markers',
                                name = names[i],
                                marker = dict(
                                    size = 2,
                                    color = colors[i],
                                    opacity = 0.6)))

plot.update_traces(hovertemplate = 'Monthly web vists: %{x} <br>Number of purchases: %{y} <br>Average Check: %{z}')

plot.update_layout(width = 800, height = 800, autosize = True, showlegend = False,
                   scene = dict(xaxis = dict(title = 'Number of web visits per month', titlefont_color = 'black'),
                                yaxis = dict(title = 'Number of web purchases', titlefont_color = 'black'),
                                zaxis = dict(title = 'Average check', titlefont_color = 'black')),
                   font = dict(family = "monospace", color  = 'black', size = 12),
                   title_text = 'Customers clusters according to Marketing Channel', title_x = 0.5)

st.plotly_chart(plot)

import plotly.express as px

fig = px.pie(data['Cluster'].value_counts().reset_index(), values='Cluster', names='index', width=700, height=700)
fig.update_traces(textposition='inside',
                  textinfo='percent + label',
                  hole=0.8,
                  marker=dict(colors=['#dd4124', '#009473', '#336b87', '#b4b4b4'], line=dict(color='white', width=2)),
                  hovertemplate='Clients: %{value}')

fig.update_layout(annotations=[dict(text='Number of clients <br>by cluster',
                                    x=0.5, y=0.5, font_size=28, showarrow=False,
                                    font_family='monospace',
                                    font_color='black')],
                  showlegend=False)

st.plotly_chart(fig)

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

st.pyplot(fig)

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

st.pyplot(fig)