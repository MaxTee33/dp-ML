import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
from sklearn.preprocessing import StandardScaler
import warnings

st.title('🤯 Wastewater Treatment Plants')
st.info('Clustering Energy Consumption Profiles')

df = pd.read_csv('https://raw.githubusercontent.com/MaxTee33/dp-ML/refs/heads/master/Data-Melbourne_F_fixed.csv')
