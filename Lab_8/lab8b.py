import matplotlib.pyplot as plt
# Importing the libraries
import pandas as pd
import numpy as np
# Import subplotspec for subplots
from matplotlib import gridspec

# Set the display properties of the dataframe
pd.options.display.max_columns = 100
pd.options.display.precision = 2

# Get the firstline as a header
cols = pd.read_csv('lab8/Log_NB_4.csv', nrows=0).columns.tolist()

# Read the dataset into a dataframe
df = pd.read_csv('lab8/Log_NB_4.csv', names=cols, skiprows=[0])

# understand  the dataset
print(df.shape)

# Get the Data types of the columns
types = df.dtypes
print('Data types of the columns: ')
print(types)

# Find the correlation between the features
corr = df.corr(method='pearson')
print('Correlation between the features: ')
print(corr)

# Get the top 5 features with the highest correlation
top5 = corr.nlargest(5, 'sport')
print('Top 5 features with the highest correlation: ')
print(top5)

# Plot the correlation matrix as a heatmap without seaborn
fig = plt.figure(figsize=(10, 10))
gs = gridspec.GridSpec(1, 1)
ax = plt.subplot(gs[0, 0])
cax = ax.matshow(corr, vmin=-1, vmax=1)
fig.colorbar(cax)
ticks = np.arange(0, len(corr.columns), 1)
ax.set_xticks(ticks)
plt.xticks(rotation=90)
ax.set_yticks(ticks)
ax.set_xticklabels(corr.columns)
ax.set_yticklabels(corr.columns)
plt.show()

# Pick and print the top values from the correlation matrix
print('Top values from the correlation matrix: ')
print(corr['sport'].nlargest(5))

