import streamlit as st
import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, AffinityPropagation, MeanShift
from sklearn.metrics import silhouette_score
import hdbscan
import optics
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('https://raw.githubusercontent.com/MaxTee33/dp-ML/refs/heads/master/processed_data.xls')
X = df.drop('Avg_Outflow', axis=1)  # Assuming 'Avg_Outflow' is the target
X_scaled = StandardScaler().fit_transform(X)  # Scaling for clustering algorithms

# Define clustering algorithms
clustering_algorithms = {
    "HDBSCAN": hdbscan.HDBSCAN(),
    "OPTICS": optics.OPTICS(),
    "Agglomerative Clustering": AgglomerativeClustering(),
    "Affinity Propagation": AffinityPropagation(),
    "Self-Organizing Maps (SOM)": 'SOM placeholder',  # To be implemented separately
    "Mean Shift": MeanShift()
}

# Function to evaluate clustering
def evaluate_clustering(X, algorithm, name):
    model = algorithm
    model.fit(X)
    labels = model.labels_

    # Calculate silhouette score (if possible)
    if len(set(labels)) > 1:
        silhouette = silhouette_score(X, labels)
    else:
        silhouette = "Not enough clusters"

    return labels, silhouette

# Streamlit User Interface to choose algorithm
st.title('Clustering Energy Consumption Profiles')
st.write("Choose a clustering algorithm:")
selected_algorithm = st.selectbox("Select Clustering Algorithm", list(clustering_algorithms.keys()))

# Run the selected clustering model
if selected_algorithm != "Self-Organizing Maps (SOM)":  # SOM requires additional implementation
    algorithm = clustering_algorithms[selected_algorithm]
    labels, silhouette = evaluate_clustering(X_scaled, algorithm, selected_algorithm)
    st.write(f"Clustering results for {selected_algorithm}:")
    st.write(labels)  # Show cluster labels
    st.write(f"Silhouette Score: {silhouette}")
else:
    st.write("Self-Organizing Maps (SOM) implementation required")

# Display results in a humanized manner
st.write("Clustering evaluation metrics:")
st.write("Performance Metrics like Silhouette Score can be used to tune and optimize the model")


st.write("Tune Hyperparameters for Clustering Algorithms:")

if selected_algorithm == "Agglomerative Clustering":
    n_clusters = st.slider("Number of clusters", 2, 20, 5)
    algorithm = AgglomerativeClustering(n_clusters=n_clusters)
elif selected_algorithm == "Affinity Propagation":
    damping = st.slider("Damping (0 to 1)", 0.5, 1.0, 0.9)
    algorithm = AffinityPropagation(damping=damping)
# Add similar customization for other algorithms
