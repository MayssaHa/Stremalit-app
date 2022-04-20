import streamlit as st
from streamlit_multipage import MultiPage
import pickle
import pandas as pd
import numpy as np

loaded_model = pickle.load(open('pages/finalized_model.sav', 'rb'))


def food(model):
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


    result= model.predict(data)
    resultproba = model.predict_proba(data)

    st.subheader('Class labels and their corresponding index number')
    st.write(['Refused Marketing Campaign', 'Accepted marketing Campaign'])

    st.subheader('Prediction results in probability')
    st.write(resultproba)
    print(result)
food(loaded_model)