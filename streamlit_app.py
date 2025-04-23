import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import warnings
warnings.filterwarnings("ignore")

from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, MeanShift, OPTICS
import hdbscan
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



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
numeric_features = ['Avg_Outflow', 'Avg_Inflow', 'Energy_Cons', 'Ammonia', 'BOD', 'COD','TN', 'Avg_Temperature', 'Max_Temperature', 'Min_Temperature', 'Avg_Humidity']
categorical_features = ['Year', 'Month', 'Day']


with st.sidebar:
  st.header('Input Features')
  with st.expander('Numeric Features'):
    Avg_Outflow = st.select_slider('Average Outflow', options=sorted(df['Avg_Outflow'].unique()))
    Avg_Inflow = st.select_slider('Average Inflow', options=sorted(df['Avg_Inflow'].unique()))
    Energy_Cons = st.select_slider('Energy Consumption', options=sorted(df['Energy_Cons'].unique()))
    Ammonia = st.select_slider('Ammonia', options=sorted(df['Ammonia'].unique()))
    BOD = st.select_slider('BOD', options=sorted(df['BOD'].unique()))
    COD = st.select_slider('COD', options=sorted(df['COD'].unique()))
    TN = st.select_slider('TN', options=sorted(df['TN'].unique()))
    Avg_Temperature = st.select_slider('Average Temperature', options=sorted(df['Avg_Temperature'].unique()))
    Max_Temperature = st.select_slider('Max Temperature', options=sorted(df['Max_Temperature'].unique()))
    Min_Temperature = st.select_slider('Min Temperature', options=sorted(df['Min_Temperature'].unique()))
    Avg_Humidity = st.select_slider('Average Humidity', options=sorted(df['Avg_Humidity'].unique()))
    
  with st.expander('Categories Features'):
    Year = st.slider('Year', 2014, 2019)
    Month = st.slider('Month', 1, 12)
    Day = st.slider('Day', 1, 31)
    
data = {'Avarage Outflow': [Avg_Outflow],
        'Average Inflow' : [Avg_Inflow],
        'Energy Consumption' : [Energy_Cons],
        'Ammonia' : [Ammonia],
        'BOD' : [BOD],
        'COD' : [COD],
        'TN' : [TN],
        'Average Temperature' : [Avg_Temperature],
        'Max Temperature' : [Max_Temperature],
        'Min Temperature' : [Min_Temperature],
        'Average Humidity' : [Avg_Humidity]
       }


scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[numeric_features])

# Apply HDBSCAN clustering
model = hdbscan.HDBSCAN(min_cluster_size=2)  # You can adjust min_cluster_size
model.fit(df_scaled)

# Get the clustering labels
cluster_labels = model.labels_

# Create a new DataFrame showing only the Cluster labels
cluster_output = pd.DataFrame({'Cluster': cluster_labels})
cluster_output





