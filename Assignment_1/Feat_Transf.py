import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn import svm, naive_bayes, metrics
from warnings import simplefilter
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectPercentile, SelectKBest, RFE, VarianceThreshold, mutual_info_classif, chi2
from sklearn.pipeline import make_pipeline
from collections import Counter as count
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.preprocessing import MinMaxScaler, StandardScaler, minmax_scale, normalize
from sklearn.linear_model import LogisticRegression

pd.options.display.width = 100
pd.options.display.precision = 2

# Get header names from the dataset A1_Mangesh/ClaMP_Raw-5184.csv
cols = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', nrows=0).columns.tolist()

# Print cols
#print(cols)

# Read the Dataset A1_Mangesh/ClaMP_Raw-5184.csv
dataset = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', names=cols, skiprows=[0])

# Print the Dataset
print(dataset.shape)

# Convert the dataset A1_Mangesh/ClaMP_Raw-5184.csv to an array
#array = dataset.values
dataset = dataset.fillna(1)

# Split the dataset of A1_Mangesh/ClaMP_Raw-5184.csv into input and output
X=dataset.drop(['CreationYear', 'NumberOfSections', 'class'], axis=1)
y=dataset['class']

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
#print(counter)

# Transform the base data.csv using SMOTE to increase the number of samples in the malware class
rand_undersample = RandomUnderSampler(random_state=0)
X_train, y_train = rand_undersample.fit_resample(X, y)
print("After undersampling: \n", count(y_train))

# Transform the data.csv file with normalize to increase increase the number of samples of the minority class (malware)
std = StandardScaler()
X_train = std.fit_transform(X_train)
print("Value distribution after Standardized: \n", count(y_train))

# Transform the data.csv file with normalize to increase the number of samples of the minority class (malware)
minmax_scale = MinMaxScaler()
X_train = minmax_scale.fit_transform(X_train)
print("Value distribution after Standardized: \n", count(y_train))

