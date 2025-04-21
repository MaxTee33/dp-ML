import streamlit as st
import hdbscan
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np

st.title('🤯 Wastewater Treatment Plants')
st.info('Clustering Energy Consumption Profiles')



with st.expander('Data'):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/MaxTee33/dp-ML/refs/heads/master/processed_data.xls')
  st.write(df.head())  # Display the first few rows of the data

  st.write('**X**')
  X = df.drop('Avg_Outflow', axis=1)
  st.write(X)

  st.write('**Y**')
  y = df.Avg_Outflow
  st.write(y)

# Define numerical and categorical features
numeric_features = [
    'Avg_Outflow', 'Avg_Inflow', 'Energy_Cons', 'Ammonia', 'BOD', 'COD',
    'TN', 'Avg_Temperature', 'Max_Temperature', 'Min_Temperature', 'Avg_Humidity'
]
categorical_features = ['Year', 'Month', 'Day']

with st.sidebar:
  st.header('Input Features')
  with st.expander('Numeric Features'):
    options = list(df['Avg_Outflow'].unique())
    options.sort()
    Avg_Outflow = st.select_slider('Average Outflow', options=options)
    
  with st.expander('Categories Features'):
    Year = st.slider('Year', 2014, 2019)
    Month = st.slider('Month', 1, 12)
    Day = st.slider('Day', 1, 31)

with st.expander('HDBSCAN'):
  scaler = StandardScaler()
  df_scaled = scaler.fit_transform(df[numeric_features])
  model = hdbscan.HDBSCAN(min_cluster_size=2)  # You can adjust min_cluster_size
  model.fit(df_scaled)
  cluster_labels = model.labels_
  cluster_output = pd.DataFrame({'Cluster': cluster_labels})
  cluster_output


