# Import the required modules
from pandas import read_csv
from pandas import set_option
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from collections import Counter
from sklearn.metrics import silhouette_score

# Set the display properities
pd.options.display.width = 100
pd.options.display.precision = 2

# Get the firstline as header
cols = read_csv(
    'Semester-5\Data_Analysis Project\Lab_7\Dataset_CS\Cyber Security Breaches.csv', nrows=0).columns.tolist()

# Read the rest of the file
dataset = read_csv(
    'Semester-5\Data_Analysis Project\Lab_7\Dataset_CS\Cyber Security Breaches.csv', names=cols, skiprows=[0])

data = dataset.dropna()

# Print the details about data
# print(dataset.dtypes)
# print(dataset.shape)

data = dataset.filter(["State", "year"])
# print(data)

# Visualize data points of State vs year using Scatter Plot
plt.scatter(data['State'], data['year'], c='black')
plt.xlabel("State")
plt.ylabel("year")
plt.title("State vs year")
plt.show()

# Plot line graph
plt.plot(data['State'], data['year'], c='black')
plt.xlabel("State")
plt.ylabel("year")
plt.title("State vs year")
plt.show()

# Call the K-Means algorithm of the KMeans class from scikit-learn to perform clustering
# All other parameters will be set to default
kmeans = KMeans(n_clusters=5).fit(dataset)

# Print Cluster Information
print('Labels')
print(kmeans.labels_)
print('Centeriod')
print(kmeans.cluster_centers_)
print('Size of each cluster')
print(Counter(kmeans.labels_))
print("Interia")
print(kmeans.inertia_)

# Print data along with cluster labels
y1 = dataset.filter(["State"]).to_numpy()
y1 = y1.astype(int)
y2 = dataset.filter(["year"]).to_numpy()
y2 = y2.astype(float)
labels = kmeans.labels_

# Plot the clusters
plt.scatter(y1, y2, c=labels, cmap='rainbow')
plt.xlabel("State")
