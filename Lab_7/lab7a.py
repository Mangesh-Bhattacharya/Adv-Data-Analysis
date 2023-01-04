import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

A = 2  # Number of clusters
B = 20  # Number of iterations
# Generate random data points for each cluster
C = 100 * np.random.rand(100, 2)
print(C)  # Print the data points
wcss = []  # Initialize the WCSS list to store the WCSS values for each iteration
# Initialize the Silhouette list to store the Silhouette values for each iteration
silhouette = []
# Initialize the k_means list to store the number of clusters for each iteration
k_means = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
for i in k_means:  # Iterate through the number of clusters list
    # Call the kmeans algorithm of the KMeans class from the sklearn.cluster module
    kmeans = KMeans(n_clusters=i, random_state=0).fit(C)
    # Get the labels of the clusters assigned to each point for each iteration
    labels = kmeans.labels_

    print("Here is Inertia: ")  # Print the WCSS for each iteration
    print(kmeans.inertia_)  # Print the WCSS for each iteration
    # Append the WCSS for each iteration to the WCSS list
    wcss.append(kmeans.inertia_)
    # Append the Silhouette score for each iteration to the Silhouette list
    silhouette.append(silhouette_score(C, labels, metric='euclidean'))

    # Count the number of Data points & Cluster Centroids in each cluster for each iteration
    print("k_means: ", i, " has ", np.count_nonzero(labels == 0), " data points and ", np.count_nonzero(
        kmeans.cluster_centers_[:, 0] == kmeans.cluster_centers_[0, 0]), " cluster centroids.")

# Plot the points in the dataset and cluster centroids as scatter plots (2D)
plt.scatter(C[:, 0], C[:, 1], c=labels, s=50, cmap='viridis')
# Plot the WCSS and the Silhouette score for each k (number of clusters) in the dataset
plt.scatter(kmeans.cluster_centers_[:, 0],
            kmeans.cluster_centers_[:, 1], c='red')
plt.title("Scatter Plot of Data Points and Cluster Centroids")
plt.legend(["Data Points", "Centroids"])
plt.show()

# Plot the WCSS and the Silhouette score for each k (number of clusters) in the dataset
plt.plot(k_means, wcss)  # Plot the WCSS for each iteration without decimals
plt.xlabel('Number of Clusters')  # Label the x-axis
plt.ylabel('WCSS')  # Label the y-axis
plt.title('Number of Clusters vs. WCSS')  # Title the plot
plt.show()  # Show the plot

# Plot the Silhouette score for each iteration
plt.plot(k_means, silhouette)
plt.xlabel('Number of Clusters')  # Label the x-axis
plt.ylabel('Silhouette')  # Label the y-axis
plt.title('Number of Clusters vs. Silhouette')  # Title the plot
plt.show()  # Show the plot
