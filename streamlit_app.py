import streamlit as st
import numpy as np
import pandas as pd

st.title('🤯 Wastewater Treatment Plants')
st.info('Clustering Energy Consumption Profiles')

with st.expender("Data"):
  st.write('**Raw data**')
  df = pd.read_csv('https://raw.githubusercontent.com/MaxTee33/dp-ML/refs/heads/master/processed_data.xls')
  df

  st.write('**X**')
  X = df.drop('Avg_Outflow', axis=1)
  X

  st.write('**Y**')
  y = df.Avg_Outflow
  y
