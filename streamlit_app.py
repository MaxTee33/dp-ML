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

import scipy.cluster.hierarchy as sch


st.title('🤯 Wastewater Treatment Plants')
st.info('Clustering Energy Consumption Profiles')


df = pd.read_csv('https://raw.githubusercontent.com/MaxTee33/dp-ML/refs/heads/master/processed_data.xls')

selected_columns = ['Avg_Outflow', 'Avg_Inflow', 'Energy_Cons', 'Ammonia', 'BOD', 'COD', 'TN', 'Avg_Temperature', 'Max_Temperature', 'Min_Temperature', 'Avg_Humidity']
numerical_features = df[selected_columns]


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
  
  df = pd.DataFrame(X, columns=["PCA Component 1", "PCA Component 2"])
  df['Cluster'] = labels  # Add the cluster labels as a new column
  
  plt.figure(figsize=(12, 10))
  scatter = plt.scatter(df['PCA Component 1'], df['PCA Component 2'], c=df['Cluster'], cmap='viridis', edgecolor='k', s=100)
  plt.title(title)
  plt.xlabel('PCA Component 1')
  plt.ylabel('PCA Component 2')
  plt.colorbar(scatter, label='Cluster Label')  # Color bar to show the cluster labels
  st.pyplot(plt)


  
# Expander of Agglomerative Clustering
with st.expander('Agglomerative Clustering'):
  selection = st.multiselect("Select features", numerical_features.columns.tolist(), default=[])  # Default selects all features
  valid_selection = [col for col in selection if col in df.columns]
  
  if len(valid_selection) >= 2:
    df_selected = df[valid_selection]
    num_rows = st.slider("Select a range of number of rows", 10, len(df), len(df))  # Use the actual number of rows in df
    st.write(f"Number of rows selected: {num_rows}")
    n_clusters = st.slider("Select number of clusters", 1, 10, 3)

    # StandardScaler and PCA
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected rows
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    agg_clustering = AgglomerativeClustering(n_clusters)
    agg_labels = agg_clustering.fit_predict(X_scaled)
    st.write(X_pca, agg_labels)
    visualize_clusters(X_pca, agg_labels, 'Agglomerative Clustering')
    
  else:
    st.write("Please select more than one features to display the scatter plot.")



def visualize_dendrogram(X_scaled):
    # Compute the linkage matrix
    Z = sch.linkage(X_scaled, method='ward')
    
    # Plot the dendrogram
    plt.figure(figsize=(10, 7))
    sch.dendrogram(Z)
    plt.title("Agglomerative Clustering Dendrogram")
    plt.xlabel("Data Points")
    plt.ylabel("Distance")
    st.pyplot(plt)  # Display the plot in Streamlit

with st.expander('Agglomerative Clustering'):
    # Select features for clustering
    selection = st.multiselect("Select features", numerical_features, default=numerical_features)  # Default selects all features
    valid_selection = [col for col in selection if col in df.columns]
    
    if len(valid_selection) >= 2:
        # Filter the selected columns
        df_selected = df[valid_selection]
        
        # Allow the user to select a range of rows
        num_rows = st.slider("Select a range of number of rows", 10, len(df), len(df))  # Use the actual number of rows in df
        st.write(f"Number of rows selected: {num_rows}")
        
        # Select number of clusters
        n_clusters = st.slider("Select number of clusters", 1, 10, 3)

        # StandardScaler and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected rows
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply Agglomerative Clustering
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters)
        agg_labels = agg_clustering.fit_predict(X_scaled)

        # Displaying the PCA results and cluster labels
        st.write("PCA Transformed Data and Agglomerative Clustering Labels:", X_pca, agg_labels)

        # Visualize Clusters using PCA Scatter Plot
        visualize_clusters(X_pca, agg_labels, 'Agglomerative Clustering')

        # Visualize Dendrogram
        visualize_dendrogram(X_scaled)





