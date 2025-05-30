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
from minisom import MiniSom
from sklearn.datasets import make_blobs
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import silhouette_score, pairwise_distances

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



st.title('🤯 Wastewater Treatment Plants')
st.info('Clustering Energy Consumption Profiles')


df = pd.read_csv('https://raw.githubusercontent.com/MaxTee33/dp-ML/refs/heads/master/X_corr_filtered_df.csv')


with st.expander('Dataset'):
  st.write('**Raw data**')
  X = df.drop('Avg_Outflow', axis=1)
  st.write(X)

  
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
    selection = st.multiselect("Select features", df.columns.tolist(), default=[], key='agg_clustering')  # Added unique key
    valid_selection = [col for col in selection if col in df.columns]
    
    if len(valid_selection) >= 2:
        df_selected = df[valid_selection]
        num_rows = st.slider("Select the desired Number of Rows", 10, len(df), len(df))  # Use the actual number of rows in df
        st.write(f"Selected number of rows: {num_rows}")
        n_clusters = st.slider("Select number of clusters", 2, 6, 2)

        # StandardScaler and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected rows
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        agg_clustering = AgglomerativeClustering(n_clusters=n_clusters, linkage='ward', metric='euclidean')
        agg_labels = agg_clustering.fit_predict(X_scaled)


        visualize_clusters(X_pca, agg_labels, 'Agglomerative Clustering')
        
        # Silhouette Score
        silhouette_avg = silhouette_score(X_scaled, agg_labels)
        st.write('Silhouette Score:', silhouette_avg)

        df['Cluster Label'] = agg_labels
        cluster_summary = df.groupby('Cluster Label').describe() # Calculate descriptive statistics for each cluster
        st.write('Average of each feature per cluster', cluster_summary)
    
    else:
        st.write("Please select more than one feature to display the scatter plot.")



# Expander of Affinity Propagation
with st.expander('Affinity Propagation'):
    selection = st.multiselect("Select features", df.columns.tolist(), default=[], key='affinity_propagation')  # Added unique key
    valid_selection = [col for col in selection if col in df.columns]
  
    if len(valid_selection) >= 2:
    
      df_selected = df[valid_selection]
      num_rows = st.slider("Select the desired Number of Rows", 10, len(df), len(df))  # Use the actual number of rows in df
      st.write(f"Selected number of rows: {num_rows}")

      preference = st.multiselect("Select the desired Number of Preference",["-50", "-25", "0", "25", "50"])
          
      # StandardScaler and PCA
      scaler = StandardScaler()
      X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected rows
      pca = PCA(n_components=2)
      X_pca = pca.fit_transform(X_scaled)

      similarity_matrix = pairwise_distances(X_scaled, metric='euclidean')
      st.write(similarity_matrix)
     
      # Apply Affinity Propagation
      aff_prop = AffinityPropagation(preference=preference, damping=0.9)
      aff_prop.fit(X_scaled)
      aff_labels = aff_prop.labels_
          
      visualize_clusters(X_pca, aff_labels, 'Affinity Propagation')
      
      # Silhouette Score
      silhouette_avg = silhouette_score(X_scaled, aff_labels)
      st.write('Silhouette Score:', silhouette_avg)
      
      try:        
          df_selected['Cluster Label'] = aff_labels
          cluster_summary = df_selected.groupby('Cluster Label').describe() # Calculate descriptive statistics for each cluster
          st.write('Average of each feature per cluster', cluster_summary)
        
      except Exception as e:
          st.write("Can't Calculate Describe Summary of Data Because of Too Few Data Points!")
      
    else:
      st.write("Please select more than one feature to display the scatter plot.")



# Expender of HDBSCAN
with st.expander('HDBSCAN'):
    selection = st.multiselect("Select features", df.columns.tolist(), default=[], key='hdbscan')  # Added unique key
    valid_selection = [col for col in selection if col in df.columns]
    
    if len(valid_selection) >= 2:
        df_selected = df[valid_selection]
        num_rows = st.slider("Select the desired Number of Rows", 10, len(df), len(df))  # Use the actual number of rows in df
        st.write(f"Selected number of rows: {num_rows}")

        cluster_size = st.slider("Select the desired Cluster Size", 2, 15, 5) 
        st.write(f"Number of Cluster Size: {cluster_size}")
        
        # StandardScaler and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected rows
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply HDBSCAN
        hdb = hdbscan.HDBSCAN(cluster_size)
        hdb_labels = hdb.fit_predict(X_scaled)
        
        visualize_clusters(X_pca, hdb_labels, 'HDBSCAN')

        try:
            # Silhouette Score
            silhouette_avg = silhouette_score(X_scaled, hdb_labels)
            st.write('Silhouette Score:', silhouette_avg)
        except Exception as e:
            st.write("Can't Calculate Silhouette Score Because of Single Cluster Scenario!")    
      
        try:
            df_selected['Cluster Label'] = hdb_labels
            cluster_summary = df_selected.groupby('Cluster Label').describe() # Calculate descriptive statistics for each cluster
            st.write('Average of each feature per cluster', cluster_summary)
        except Exception as e:
          st.write("Can't Calculate Describe Summary of Data Because of Too Few Data Points!")
    
    else:
        st.write("Please select more than one feature to display the scatter plot.")




# Expander of Mean Shift
with st.expander('Mean Shift'):
    selection = st.multiselect("Select features", df.columns.tolist(), default=[], key='mean_shift')  # Added unique key
    valid_selection = [col for col in selection if col in df.columns]
    
    if len(valid_selection) >= 2:
        df_selected = df[valid_selection]
        num_rows = st.slider("Select the desired Number of Rows", 10, len(df), len(df))  # Use the actual number of rows in df
        st.write(f"Selected number of rows: {num_rows}")
      
        num_bandwidth = st.slider("Select the desired Bandwidth", 0.5, 20.0, 1.0) 
        st.write(f"Selected bandwidth selected: {num_bandwidth}")
        
        # StandardScaler and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected row
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        
        # Apply Mean Shift
        mean_shift = MeanShift(bandwidth=num_bandwidth)
        mean_shift.fit(X_scaled)
        mean_shift_labels = mean_shift.labels_
        
        visualize_clusters(X_pca, mean_shift_labels, 'Mean Shift')

        try:
            # Silhouette Score
            silhouette_avg = silhouette_score(X_scaled, mean_shift_labels)
            st.write('Silhouette Score:', silhouette_avg)
        except Exception as e:
            st.write("Can't Calculate Silhouette Score Because of Single Cluster Scenario!") 
      
        try:
            df_selected['Cluster Label'] = mean_shift_labels
            cluster_summary = df_selected.groupby('Cluster Label').describe() # Calculate descriptive statistics for each cluster
            st.write('Average of each feature per cluster', cluster_summary)
          
        except Exception as e:
          st.write("Can't Calculate Describe Summary of Data Because Too Few of Data Points!")
    
    else:
        st.write("Please select more than one feature to display the scatter plot.")




# Expander of OPTICS
with st.expander('OPTICS'):
    selection = st.multiselect("Select features", df.columns.tolist(), default=[], key='optics')  # Added unique key
    valid_selection = [col for col in selection if col in df.columns]
    
    if len(valid_selection) >= 2:
        df_selected = df[valid_selection]
        num_rows = st.slider("Select the desired Number of Rows", 10, len(df), len(df))  # Use the actual number of rows in df
        st.write(f"Selected number of rows: {num_rows}")

        min_samples = st.slider("Select the desired Minimum Samples", 5, 20, 10) 
        st.write(f"Selected minimum samples: {min_samples}")
      
        # StandardScaler and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_selected[:num_rows])  # Scale only the selected rows
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Apply OPTICS with specified parameters
        optics = OPTICS(min_samples=min_samples, xi=0.05, min_cluster_size=0.1)
        optics.fit(X_scaled)
        optics_labels = optics.labels_
        
        visualize_clusters(X_pca, optics_labels, 'OPTICS')
        
        # Silhouette Score
        if 1 < len(set(optics_labels)) < X_scaled.shape[0]:
            silhouette_avg = silhouette_score(X_scaled, optics_labels)
            st.write('Silhouette Score:', silhouette_avg)
        else:
            st.write("Cannot compute Silhouette Score due to too few valid clusters.")

        df_selected = df_selected.copy()
        df_selected['Cluster Label'] = optics_labels
        cluster_summary = df_selected.groupby('Cluster Label').describe()  # Descriptive statistics for each cluster
        st.write('Average of each feature per cluster', cluster_summary)
    
    else:
        st.write("Please select more than one feature to display the scatter plot.")



# Expander of Self-Organizing Maps (SOM)
with st.expander('Self-Organizing Maps (SOM)'):
    selection = st.multiselect("Select features", df.columns.tolist(), default=[], key='som')  # Added unique key
    valid_selection = [col for col in selection if col in df.columns]

    if len(valid_selection) >= 2:
        df_selected = df[valid_selection]
        num_rows = st.slider("Select the desired Number of Rows", 10, len(df), len(df))
        st.write(f"Selected number of rows: {num_rows}")

        # StandardScaler and PCA
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(df_selected[:num_rows])
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # Train SOM using best parameters
        som = MiniSom(15, 15, len(valid_selection), sigma=1.0, learning_rate=0.5)
        som.train(X_scaled, 300)  # 300 epochs

        # Assign cluster labels based on SOM winner nodes
        som_labels = np.array([som.winner(x)[0] * 15 + som.winner(x)[1] for x in X_scaled])

        # Visualization using PCA output
        visualize_clusters(X_pca, som_labels, 'Self-Organizing Maps (SOM)')


        # Quantization Error
        quantization_error = som.quantization_error(X_scaled)
        st.write('Quantization Error:', quantization_error)
        
        # Topographic Error
        def topographic_error(som, data):
            error_count = 0
            for x in data:
                dists = np.linalg.norm(som._weights - x, axis=2)
                sorted_indices = np.unravel_index(np.argsort(dists, axis=None), dists.shape)
                bmu1 = (sorted_indices[0][0], sorted_indices[1][0])
                bmu2 = (sorted_indices[0][1], sorted_indices[1][1])
                if abs(bmu1[0] - bmu2[0]) + abs(bmu1[1] - bmu2[1]) > 1:
                    error_count += 1
            return error_count / len(data)
        
        topo_error = topographic_error(som, X_scaled)
        st.write('Topographic Error:', topo_error)
        
        # Total Score (lower is better)
        total_score = quantization_error + topo_error
        st.write('Total Score (Quantization + Topographic Error):', total_score)


    else:
        st.write("Please select more than one feature to display the scatter plot.")





