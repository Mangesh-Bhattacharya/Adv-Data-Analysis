###########################################################################
# Name: Mangesh Bhattacharya
# Std No.: 039-251-145
# Course: SRT 521 - adv. Data analysis
# Inst.: Dr. asma Paracha
# Date: 2022-11-13
# Description: Lab 8b
# Task 3: Hierarchical Agglomerative Clustering
###########################################################################

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

# Make a scatter plot by pairing any two of them at a time, you can make at least 3 different plots to plot all the features.
# Set the display properties of the dataframe
pd.options.display.max_columns = 100
pd.options.display.precision = 2

# Get the firstline as a header
cols = pd.read_csv('lab8/Log_NB_4.csv', nrows=0).columns.tolist()

# Read the dataset into a dataframe
df = pd.read_csv('lab8/Log_NB_4.csv', names=cols, skiprows=[0])

# understand the dataset
print(df.shape)

# Pick only 5 features from the dataset "sport", "swin", "dwin", "dtcpb", and "stcpb" and plot them in a scatter plot matrix
df = df[['sport', 'swin', 'dwin', 'dtcpb', 'stcpb']]
print(df.head())

# Compare Scatter Plot
df.plot.scatter(x='sport', y='swin')
df.plot.scatter(x='sport', y='dwin')
plt.title('Scatter Plot of sport vs swin')
plt.show()

# Drop the rows by 10000
df = df.drop(df.index[10000:])
print(df.shape)
# Create an AgglomerativecClustering object and number of clusters to 2, 5, and 10
agg = AgglomerativeClustering(
    n_clusters=2, affinity='euclidean', linkage='ward')
agg = AgglomerativeClustering(
    n_clusters=5, affinity='euclidean', linkage='ward')
agg = AgglomerativeClustering(
    n_clusters=10, affinity='euclidean', linkage='ward')

# Fit the model to the data, predict the labels
agg.fit_predict(df)
agg.fit(df)
plt.figure(figsize=(20, 10))
plt.title("Hierarchical Agglomerative Clustering - Dendrogram")
dend = shc.dendrogram(shc.linkage(df, method='ward'))
# Rotate rowname to fit in vertically
plt.xticks(rotation=90, fontsize=8, ha='center')
plt.show()
