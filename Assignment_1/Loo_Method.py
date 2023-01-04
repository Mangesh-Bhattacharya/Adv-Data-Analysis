from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn import datasets, pipeline
from sklearn.metrics import plot_confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score

# Set the display properities
pd.options.display.width = 100
pd.options.display.precision = 2

# Get the firstline as header
cols = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', nrows=0).columns.tolist()

# Read the rest of the file
dataset = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', names=cols, skiprows=[0])
dataset = dataset.fillna(1)

# Split the data into train and test data
X=dataset.drop(['CreationYear', 'NumberOfSections', 'class'], axis=1)
y=dataset['class']

# Create a LeaveOneOut instance and get the indices of the training and test folds for each iteration
loo = LeaveOneOut()
loo.get_n_splits(X)
loo.split(X, y)

# Create a list to store the scores
scores = cross_val_score(DecisionTreeClassifier(), X, y, cv=loo)


print("Cross Validation Scores: ", scores)
print("Average CV Score: ", scores.mean())
print("Number of CV Scores used in Average: ", len(scores))

# Plotting histogram of the scores for LeaveOneOut cross validation
plt.hist(scores, bins=10)
plt.title("Histogram of the scores for LeaveOneOut cross validation")
plt.xlabel("Scores")
plt.ylabel("Accuracy")
plt.show()

# Save the plot
plt.savefig('A1_Mangesh/LOO_CV_Scores.png')