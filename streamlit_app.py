import streamlit as st
from sklearn.cluster import AgglomerativeClustering, DBSCAN
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
@st.cache
def load_data():
    return pd.read_csv('https://raw.githubusercontent.com/MaxTee33/dp-ML/refs/heads/master/processed_data.xls')

# Display dataset preview
st.write('**Data Preview**')
df = load_data()
st.write(df.head())

# Select clustering technique
clustering_method = st.selectbox('Select Clustering Method', ['Agglomerative Clustering', 'DBSCAN', 'HDBSCAN', 'OPTICS', 'Mean Shift'])

# Display parameter options based on selected method
if clustering_method == 'Agglomerative Clustering':
    n_clusters = st.slider('Number of Clusters', 2, 10, 5)
    model = AgglomerativeClustering(n_clusters=n_clusters)
elif clustering_method == 'DBSCAN':
    eps = st.slider('EPS', 0.1, 2.0, 0.5)
    min_samples = st.slider('Min Samples', 2, 10, 5)
    model = DBSCAN(eps=eps, min_samples=min_samples)

# Train model and show clusters
st.write('**Clustering Results**')
X = df.drop('Avg_Outflow', axis=1)  # Adjust based on your features
y = df['Avg_Outflow']
model.fit(X)
labels = model.labels_

# Show results
st.write('Cluster Labels:', labels)

# Visualize clusters (e.g., 2D scatter plot)
st.write('**Cluster Visualization**')
plt.scatter(X.iloc[:, 0], X.iloc[:, 1], c=labels)
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
st.pyplot()

# Show evaluation metrics
st.write('**Evaluation Metrics**')
# You can add custom metrics like silhouette score or others

