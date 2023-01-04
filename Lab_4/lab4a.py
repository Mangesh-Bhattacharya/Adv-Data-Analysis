import random
import pandas as pd
import numpy as np
import imblearn as imb
import pickle
from collections import Counter as count
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectPercentile, SelectKBest, RFE, mutual_info_classif, chi2
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn import metrics
from warnings import simplefilter
import matplotlib.pyplot as plt

pd.options.display.width = 100
pd.options.display.precision = 2

# Calculate the number of instances in each class are in the data.csv file
data = pd.read_csv('data.csv')
# print(data['Target'].value_counts())

# Print Counter of the number of instances in each class
# print(count(data['Target']))

# Read the Dataset
dataset = pd.read_csv('data.csv')

# Specify the features and the target
X = dataset.drop(['MD5', 'Label', 'Target'], axis=1)
y = dataset.Target

# Summarize the class distribution
print("Base class distribution: \n")
counter = count(y)
print(counter)

# Transform the base dataset using random oversampling to increase the number of samples of the minority class (malware)
#rand_oversample = RandomOverSampler(sampling_strategy='minority')
rand_oversample = RandomOverSampler(random_state=0)
X_train, y_train = rand_oversample.fit_resample(X, y)
print("After oversampling: \n", count(y_train))

# Transform the data.csv file with SMOTE to increase increase the number of samples of the minority class (malware)

random_oversample = SMOTE()
X_train, y_train = random_oversample.fit_resample(X, y)
print("Oversampling after SMOTE: \n" + str(count(y_train)))

# Summarize the new class distribution
#counter = count(y)
# print(counter)

# Transform the base data.csv using SMOTE to increase the number of samples in the malware class
rand_undersample = RandomUnderSampler(random_state=0)
X_train, y_train = rand_undersample.fit_resample(X, y)
print("After undersampling: \n", count(y_train))
