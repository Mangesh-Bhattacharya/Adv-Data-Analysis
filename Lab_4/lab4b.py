import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn import svm, naive_bayes, metrics
from warnings import simplefilter
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, SelectKBest, RFE, mutual_info_classif, chi2

pd.options.display.width = 100
pd.options.display.precision = 2

# Read the Dataset
dataset = pd.read_csv('data.csv')

# Specify the features and the target
X = dataset.drop(['MD5', 'Label', 'Target'], axis=1)
y = dataset.Target

# Split the dataset into training, testing sets and stratify the split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0, stratify=y)

# Select the best features using SelectKBest
indicator = SelectKBest(chi2, k=15)
X_train = indicator.fit_transform(X_train, y_train)
print(X.shape)
print(X_train.shape)
print("Selected Features: ", indicator.get_support(indices=True))

# Call the selectKBest for selecting features using selectPercentile and chi2.
indicator = SelectPercentile(score_func=chi2, percentile=40)
X_train = indicator.fit_transform(X_train, y_train)
print(X.shape)
print(X_train.shape)
# Print the list of features selected
print("Selected Features: ", indicator.get_support(indices=True))
print("Types of Features: ", indicator.get_support())
print("Predicted Features: ", indicator.get_support(indices=True))
print("Feature Selection method: ", indicator.get_params())

#numerical_features = ['Types of Features', 'Predicated/Categorial Features', 'Feature Selection Method', indicator.get_support(indices=True), indicator.get_support(), indicator.get_params()]
#categorical_features = ['Types of Features', 'Predicted Features', 'Feature Selection Method', indicator.get_support(indices=True), indicator.get_support(), indicator.get_params()]
#numerical_features = X.select_dtypes(include=[np.number])
#categorical_predictions = X.select_dtypes(exclude=[np.number])
#print("Numerical Features: ", numerical_features.shape)
#print("Categorical Features: ", categorical_predictions.shape)

# Create a list of Numerical and Categorical Features and their names
numerical_features = X.select_dtypes(include=[np.number])
categorical_features = X.select_dtypes(exclude=[np.number])
numerical_features_names = numerical_features.columns
categorical_features_names = categorical_features.columns
print("Numerical Features: ", numerical_features_names, numerical_features.shape)
print("Categorical Features: ", categorical_features_names,
      categorical_features.shape)

# Use recursive feature elimination to select 15 features using SVC classifier
svc = svm.SVC(kernel='linear', C=1)
rfe = RFE(estimator=svc, n_features_to_select=15, step=1)
X_train = rfe.fit_transform(X_train, y_train)

# Print the list of features selected
print("Selected Features: ", rfe.get_support(indices=True))
print("Types of Features: ", rfe.get_support())
print("Predicted Features: ", rfe.get_support(indices=True))
print("Feature Selection method: ", rfe.get_params())
