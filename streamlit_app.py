import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import missingno as msno
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")

from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, MeanShift, OPTICS
import hdbscan
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


st.title('🤯 Wastewater Treatment Plants')
st.info('Clustering Energy Consumption Profiles')

df = pd.read_csv('https://raw.githubusercontent.com/MaxTee33/dp-ML/refs/heads/master/processed_data.xls')
# Define numerical and categorical features
numeric_features = ['Avg_Outflow', 'Avg_Inflow', 'Energy_Cons', 'Ammonia', 'BOD', 'COD','TN', 'Avg_Temperature', 'Max_Temperature', 'Min_Temperature', 'Avg_Humidity']
categorical_features = ['Year', 'Month', 'Day']


with st.expander('Data'):
  st.write('**Raw data**')
  st.write(df.head())  # Display the first few rows of the data

  st.write('**X**')
  X = df.drop('Avg_Outflow', axis=1)
  st.write(X)

  st.write('**Y**')
  y = df.Avg_Outflow
  st.write(y)




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


  
  # Function to visualize clustering results with color map
  def visualize_clusters(X, labels, title):
      # Create a DataFrame for easier handling
      df = pd.DataFrame(X, columns=["PCA Component 1", "PCA Component 2"])
      df['Cluster'] = labels  # Add the cluster labels as a new column
      
      # Set up the plot with a color map
      plt.figure(figsize=(12, 10))
      scatter = plt.scatter(df['PCA Component 1'], df['PCA Component 2'], c=df['Cluster'], cmap='viridis', edgecolor='k', s=100)
      plt.title(title)
      plt.xlabel('PCA Component 1')
      plt.ylabel('PCA Component 2')
      plt.colorbar(scatter, label='Cluster Label')  # Color bar to show the cluster labels
      st.pyplot(plt)  # Display the plot in Streamlit





with st.expander('Clusters'):
  # List of options for feature selection
  options = ['Avg_Outflow', 'Avg_Inflow', 'Energy_Cons', 'Ammonia', 'BOD', 'COD','TN', 'Avg_Temperature', 'Max_Temperature', 'Min_Temperature', 'Avg_Humidity']
  
  selection = st.multiselect("Select features", options, default=options)
  valid_selection = [col for col in selection if col in df.columns]
  X = df[valid_selection]
  num_entries = X.shape[0]
  
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(X)
  pca = PCA(n_components=3)
  X_pca = pca.fit_transform(X_scaled)
  
  if __name__ == "__main__":
      # 1. Generate sample data
      np.random.seed(42)
      X = np.random.randn(1000, num_entries)
      
      # 2. Scale the data
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(X)
      
      # 3. Perform PCA to reduce to 2 components (for visualization)
      pca = PCA(n_components=2)
      X_pca = pca.fit_transform(X_scaled)
      
      # 4. Perform Agglomerative Clustering
      agg_clustering = AgglomerativeClustering(n_clusters=6)  # 3 clusters
      agg_labels = agg_clustering.fit_predict(X_scaled)  # Cluster labels
      
      # 5. Visualize the clustering results using the function
      visualize_clusters(X_pca, agg_labels, 'Agglomerative Clustering')





