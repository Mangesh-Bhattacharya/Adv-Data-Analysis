import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.feature_selection import SelectKBest, chi2

# Read the data
data = pd.read_csv('data.csv')
# create the classification model selected one with default parameter and use it on training and testing data sets
X = data.iloc[:, 0:30]
y = data.iloc[:, 30]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)
model = SVC()
model.fit(X_train, y_train)
print('Accuracy of the model on training data is: ',
      model.score(X_train, y_train))
print('Accuracy of the model on testing data is: ', model.score(X_test, y_test))

#
