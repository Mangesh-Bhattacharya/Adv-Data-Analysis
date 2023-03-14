import numpy as np
from pandas import read_csv
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, mutual_info_classif
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# Read the dataset from data.csv
cols = read_csv('data.csv', nrows=0).columns.tolist()
# Read all row in the csv file and save the data in var data
dataset = read_csv('data.csv', names=cols, skiprows=[0])
# Convert the data into numpy array
array = dataset.values
# Split the data into train and test data
X = array[:, 3:]
y = dataset['Target']

# List to store names of the methods used for feature selection
titles = []
# List to store the selected features
cases = []

# Add each feature selection method to the list called Titles
titles.append("SKBt_Chi2")
# Add each output of the feature selection type into the list called Cases
cases.append(SelectKBest(chi2, k=15))
# Only 15 features are selected from the dataset

titles.append("SKBt_Mutual_Info")
cases.append(SelectKBest(mutual_info_classif, k=15))

titles.append("SPt_Mutual_Info")
cases.append(SelectPercentile(mutual_info_classif))

titles.append("SPt_Chi2")
cases.append(SelectPercentile(chi2))

# Create a pipeline to send the data to the classifier
SVC_classif = SVC(kernel='linear', C=1)

for m, n in zip(titles, cases):
    # Create a pipeline to send the data to the classifier
    print(m)
    # Create a pipeline to send the data to the classifier
    print(n)
    # Create a pipeline to send the data to the classifier and then to the feature selection method
    pipeline = make_pipeline(n, SVC_classif)
    # Perform Stratified K-Fold to split the dataset into 10 folds
    skfold = StratifiedKFold(n_splits=10).split(X, y)
    # Calculate the mean of the cross validation scores
    scores = cross_val_score(pipeline, X, y, scoring='f1', cv=skfold)
    # Print the mean of the cross validation scores and the name of the feature selection method
    print("Mean: \t", scores.mean())
    print("Standard Deviation: ", scores.std())
