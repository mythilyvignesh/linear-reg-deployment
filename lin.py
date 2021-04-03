#!/usr/bin/env python
# coding: utf-8

# In[2]:


import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
st.title('model deployement:linear regression')
st.sidebar.header('User input parameter')
def user_input_features():
    Avg_Session_Length = st.sidebar.number_input(" Avg_Session_Length")
    Time_on_App = st.sidebar.number_input("Time_on_App")
    Time_on_Website = st.sidebar.number_input("Time_on_Websie")
    Length_of_Membership = st.sidebar.number_input("Length_of_Membership")     
    data = {'Avg_Session_Length':Avg_Session_Length,
           'Time_on_App':Time_on_App,
           'Time_on_Website':Time_on_Website,
           'Length_of_Membership':Length_of_Membership,
           }
    features=pd.DataFrame(data,index=[0])
    return features
df=user_input_features()
st.subheader('User input parameters')
st.write(df)
A= pd.read_csv ('D:\\jup notebook\\ml\\e.csv')
A.drop(['Email','Address','Avatar'],axis=1,inplace=True)
A=A.dropna()
x = A[['Avg. Session Length', 'Time on App', 'Time on Website','Length of Membership']]
y = A[['Yearly Amount Spent']]
logmodel = LinearRegression()
logmodel.fit(x,y)
predictions = logmodel.predict(df)
st.subheader('Yearly Amount Spent')
st.write(predictions)


# In[ ]:





# In[ ]:




