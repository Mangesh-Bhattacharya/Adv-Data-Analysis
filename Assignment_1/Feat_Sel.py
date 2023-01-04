import array
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn import svm, naive_bayes, metrics
from warnings import simplefilter
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, SelectKBest, RFE, VarianceThreshold, mutual_info_classif, chi2
from sklearn.pipeline import make_pipeline

pd.options.display.width = 100
pd.options.display.precision = 2

# Get header names from the dataset A1_Mangesh/ClaMP_Raw-5184.csv
cols = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', nrows=0).columns.tolist()

# Print cols
#print(cols)

# Read the Dataset A1_Mangesh/ClaMP_Raw-5184.csv
dataset = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', names=cols, skiprows=[0])

# Print the Dataset
#print(dataset.shape)

# Convert the dataset A1_Mangesh/ClaMP_Raw-5184.csv to an array
#array = dataset.values

# Split the dataset of A1_Mangesh/ClaMP_Raw-5184.csv into input and output
X=dataset.drop(['CreationYear', 'NumberOfSections', 'class'], axis=1)
Y=dataset['class']

#X = array[:, 2:245]
#Y = array[:, 2]

# Split the dataset of A1_Mangesh/ClaMP_Raw-5184.csv into training and testing and use the stratified train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.90, random_state=0, stratify=Y)

# Remove features with low variance and various options are mentioned below
# Threshold values are 0.80, 0.85, 0.90, 0.95
# VarianceThreshold(threshold=(threshold * (1 - threshold)))
#indicator = VarianceThreshold(threshold=(0.95*(1-0.95)))
indicator = VarianceThreshold(threshold=(0.90*(1-0.90)))
#indicator = VarianceThreshold(threshold=(0.85*(1-0.85)))
#indicator = VarianceThreshold(threshold=(0.80*(1-0.80)))
# Fit the indicator to the training data
indicator.fit(X_train)
# Get the indices of the features that are not constant
indices = indicator.get_support()

# Print the indices of the features that are not constant
print(indices)

# Print the number of features that are not constant
print(indices.sum())

# Print the number of features that are constant
print(len(indices) - indices.sum())

# Print the features that are not constant
print(X_train.columns[indices])

# Print the features that are constant
print(X_train.columns[~indices])

# Print the shape of the training data
print(X_train.shape)

# Print the shape of the testing data
print(X_test.shape)

# Print the shape of the training data
print(Y_train.shape)

# Print the shape of the testing data
print(Y_test.shape)

# Feature Transformation using confusion matrix for dataset A1_Mangesh/ClaMP_Raw-5184.csv
# Print the confusion matrix for dataset A1_Mangesh/ClaMP_Raw-5184.csv
print(metrics.confusion_matrix(Y_test, Y_train))

# Print the accuracy score for dataset A1_Mangesh/ClaMP_Raw-5184.csv
print(metrics.accuracy_score(Y_test, Y_train))

# Print the classification report for dataset A1_Mangesh/ClaMP_Raw-5184.csv
print(metrics.classification_report(Y_test, Y_train))