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
from sklearn.metrics import silhouette_score

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



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


# Expander of Affinity Propagation
with st.expander('Affinity Propagation'):
    selection = st.multiselect("Select features", numerical_features.columns.tolist(), default=[], key='agg_clustering')  # Default selects all features
    valid_selection = [col for col in selection if col in df.columns]
    
    if len(valid_selection) >= 2:
        df_selected = df[valid_selection]
        num_rows = st.slider("Select a range of number of rows", 10, len(df), len(df))  # Use the actual number of rows in df
        st.write(f"Number of rows selected: {num_rows}")
        
        # StandardScaler and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected rows
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply Affinity Propagation
        aff_prop = AffinityPropagation(preference=-50)
        aff_prop.fit(X_scaled)
        aff_labels = aff_prop.labels_
        
        visualize_clusters(X_pca, aff_labels, 'Affinity Propagation')
        
        # Silhouette Score
        silhouette_avg = silhouette_score(X_scaled, aff_labels)
        st.write('Silhouette Score:', silhouette_avg)

        df_selected['Cluster Label'] = aff_labels
        cluster_summary = df_selected.groupby('Cluster Label').describe() # Calculate descriptive statistics for each cluster
        st.write('Average of each feature per cluster', cluster_summary)
    
    else:
        st.write("Please select more than one feature to display the scatter plot.")



# Expander of Affinity Propagation
with st.expander('Affinity Propagation'):
    selection = st.multiselect("Select features", numerical_features.columns.tolist(), default=[], key='affinity_propagation')  # Added unique key
    valid_selection = [col for col in selection if col in df.columns]
  
    if len(valid_selection) >= 2:
    
      df_selected = df[valid_selection]
      num_rows = st.slider("Select a range of number of rows", 10, len(df), len(df))  # Use the actual number of rows in df
      st.write(f"Number of rows selected: {num_rows}")
          
      # StandardScaler and PCA
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected rows
      pca = PCA(n_components=2)
      X_pca = pca.fit_transform(X_scaled)
          
      # Apply Affinity Propagation
      aff_prop = AffinityPropagation(preference=-50)
      aff_prop.fit(X_scaled)
      aff_labels = aff_prop.labels_
          
      visualize_clusters(X_pca, aff_labels, 'Affinity Propagation')
          
      # Silhouette Score
      silhouette_avg = silhouette_score(X_scaled, aff_labels)
      st.write('Silhouette Score:', silhouette_avg)
  
      df_selected['Cluster Label'] = aff_labels
      cluster_summary = df_selected.groupby('Cluster Label').describe() # Calculate descriptive statistics for each cluster
      st.write('Average of each feature per cluster', cluster_summary)
      
    else:
      st.write("Please select more than one feature to display the scatter plot.")










