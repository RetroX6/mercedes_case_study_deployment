
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import OrdinalEncoder
import xgboost as xgb
from xgboost.sklearn import XGBRegressor
from sklearn.metrics import r2_score
import joblib
import warnings
warnings.filterwarnings("ignore") 

st.title('Mercedes Benz Greener Manufacturing : Predicting the Test Time of a Mercedes Benz', anchor=None)
st.header('Objective')
st.subheader('The model will predict the test time of each mercedes benz spent on a test bench')

@st.cache
def load_data(nrows):
    data = pd.read_csv('train.csv', nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    return data

data_load_state = st.text('Loading data...')
# Load 10,000 rows of data into the dataframe.
data = load_data(50)
# Notify the reader that the data was successfully loaded.
data_load_state.text("Loading Data...Done! (using st.cache)")

if st.checkbox('Show Raw Data'):
    st.subheader('Raw data')
    st.write(data)