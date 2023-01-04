###########################################################################
# Name: Mangesh Bhattacharya
# Std No.: 039-251-145
# Course: SRT 521 - Adv. Data Analysis
# Inst.: Dr. Asma Paracha
# Date: 2022-10-01
###########################################################################
from statistics import variance
import numpy as np
import pandas as pd
import graphviz
from pandas import array, set_option
from pandas import read_csv
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn import datasets, tree
from sklearn.tree import export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from matplotlib import pyplot as plt
from lib2to3.pgen2.pgen import DFAState

# Task 1
# Display Properties of the Dataset
pd.options.display.width = 100
pd.options.display.precision = 2

# Get header names from the dataset
cols = pd.read_csv('dataset.csv', nrows=0).columns.tolist()

# Read the Dataset
dataset = pd.read_csv('dataset.csv', names=cols, skiprows=[0])
# Print the Dataset
# print(dataset)

# Need multiple columns to be dropped
dataset.drop(labels=['MD5', 'label', 'Target'], axis=1, inplace=True)
#DFAState.drop(labels=['MD5', 'label', 'Target'], axis=1, inplace=True)

# Convert the dataset to an array
array = dataset.values
# Split the dataset into input and output
X = array[:, 0:293]
Y = array[:, 0:2]

# Split the dataset into training and testing and use the stratified train-test split
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.50, random_state=1, stratify=Y)

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
indices = indicator.get_support(indices=True)
# Get the names of the features that are not constant
features = dataset.columns[indices]
# Print the names of the features that are not constant
print(features)
#dataset = indicator.fit_transform(X_train)
print(indicator)

# Task 2
# Create a Decision Tree Classifier
classifier = tree.DecisionTreeClassifier(random_state=0)
# Fit the classifier to the training data and labels
classifier = tree.DecisionTreeClassifier(random_state=0, max_depth=5)
classifier = tree.DecisionTreeClassifier(criterion="entropy", random_state=0)

# Model training and prediction
classifier = classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)  # Predict the labels of the test data
# Fit the classifier to the training data and labels
classifier.fit(X_train, Y_train)

# Evaluating the performance and accuracy of the model
print(confusion_matrix(Y_test, Y_pred))
print(classification_report(Y_test, Y_pred))
print(accuracy_score(Y_test, Y_pred))

tree.plot_tree(classifier)  # Plot the decision tree classifier
plt.show()  # Show the plot of the decision tree classifier

# Print the decision tree in image format
dot_data = tree.export_graphviz(
    classifier, out_file=None, filled=True, rounded=True, special_characters=True)
# Create graph from dot data and save it in a file named tree.png
graph = graphviz.Source(dot_data)
# Render the graph in png format and save it in a file named dataset
graph.render("dataset", format="png")
# Render the graph in png format and save it in a file with entropy
graph.render("dataset-entropy", format="png")
# Render the graph in png format and save it in a file with depth 5
graph.render("dataset-depth-5", format="png")

# load dataset into a text file and print a report
range = export_text(classifier, feature_names=cols[0:293])
f = open("report.txt", "w")
f.write(range)
f.close()
