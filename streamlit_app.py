import streamlit as st
import pandas as pd
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
  numeric_features = [
      'Avg_Outflow', 'Avg_Inflow', 'Energy_Cons', 'Ammonia', 'BOD', 'COD',
      'TN', 'Avg_Temperature', 'Max_Temperature', 'Min_Temperature', 'Avg_Humidity'
  ]
  categorical_features = ['Year', 'Month', 'Day']

  st.write('**Numeric Features**')
  st.write(numeric_features)

  st.write('**Categorical Features**')
  st.write(categorical_features)


# Select numerical columns (both float and integer types) from df_no_outliers
numeric_cols = df_no_outliers.select_dtypes(include=['float64', 'int64']).columns

# Apply StandardScaler to the numerical columns of df_no_outliers
df_scaled = StandardScaler().fit_transform(df_no_outliers[numeric_cols])

# Convert the scaled values back into a DataFrame for easier handling
df_scaled = pd.DataFrame(df_scaled, columns=numeric_cols)

pca = PCA(n_components=11)  # Try using 5 components
df_pca = pca.fit_transform(df_scaled)
cumulative_variance = pca.explained_variance_ratio_.cumsum()
print("Cumulative Explained Variance with 11 components:", cumulative_variance)
