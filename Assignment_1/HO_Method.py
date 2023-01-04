import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold, GridSearchCV, RandomizedSearchCV, learning_curve, cross_validate, LeaveOneOut, cross_val_score
from sklearn.tree import DecisionTreeClassifier

# Read the dataset from A1_Mangesh/ClaMP_Raw-5184.csv
cols = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', nrows=0).columns.tolist()
# Read all row in the csv file and save the data in var data
dataset = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', names=cols, skiprows=[0])
dataset = dataset.dropna(1)

# Split the data into train and test data
X=dataset.drop(['CreationYear', 'NumberOfSections', 'class'], axis=1)
y=dataset['class']

# Split the array into random train and test subsets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

print(len(X_train), len(X_test), len(y_train), len(y_test))

# Create a Decision Tree Classifier
clf = DecisionTreeClassifier(random_state=0)

# Train the model using the training sets
clf.fit(X_train, y_train)

# Predict the response for test dataset
y_pred = clf.predict(X_test)

# Model Accuracy, how often is the classifier correct?
print("Accuracy:",metrics.accuracy_score(y_test, y_pred))

# Print Cross Validation Score
scores = cross_val_score(clf, X, y, cv=10)
print("Cross Validation Score: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

