# importing modules
from sklearn.feature_selection import SelectKBest, chi2, SelectPercentile, mutual_info_classif
import numpy as np
from pandas import read_csv
import numpy
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_classif
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline

# TASK-1
# reading the data from dataset.csv
# saving the first line of the file, as cols variable
cols = read_csv('Semester-5\Data_Analysis Project\Lab_5\data.csv',
                nrows=0).columns.tolist()
# reading all the rows in the file and saving the data in the variable data
data = read_csv('Semester-5\Data_Analysis Project\Lab_5\data.csv',
                names=cols, skiprows=[0])

# converting the data frame object into an array
array = data.values

# dataset is saved in variable X after removing the first 3 columns(md5hash, label and target)
# splitting the data so that all the features are in X and the target is in y
X = array[:, 3:]
# I kept having ValueError for variable y, even after trying to change the datatype to int, so I opted this solution to assing the target to y
y = data['Target']
# using StratifiedKFold to split the data in 10 folds
strf = StratifiedKFold(n_splits=10).split(X, y)
# splitting the data into train and test subsets
#strf = strf.split(X,y)

# creating a list to store the name of the method used
titles = []
# creating another list to store the selected features
cases = []

# adding each feature selection strategy into the list 'titles'
titles.append("SelectKBest Chi2")
# and adding the results of each feature selection type into the list 'cases'
cases.append(SelectKBest(chi2, k=10))
# the value of k is 10 so only 10 features get selected

titles.append("SelectKBest Mutual info")
cases.append(SelectKBest(mutual_info_classif, k=10))

titles.append("SelectPercentile Mutual info")
cases.append(SelectPercentile(score_func=mutual_info_classif))
# the default percentile value is 10%, so only 10% features get selected

# we need one more feature select strategy to complete the table in question 2 of the lab
titles.append("SelectPercentile Chi2")
cases.append(SelectPercentile(score_func=chi2))

# printing the lists to verify the results
#print("titles:\n", titles)
#print("cases:\n", cases)

# sending the two lists to the classifier model
classifier = SVC(kernel='linear')

# creating a loop, and calling the function cross_val_score
for i, j in zip(titles, cases):
    # creating the pipeline that selects features then trains a support vector classifier
    # classifier will be same for all the feature selection methods
    # printing the feature selection method
    print(i)
    # creating the pipeline
    print(j)
    pipeline = make_pipeline(j, classifier)
    # cv value is the StratifiedKFold with 10 folds, cv=10 represents the same
    # passing instance of pipeline and training and test data set
    scores = cross_val_score(pipeline, X=X,
                             y=y, scoring="f1", cv=strf)
    # since we need f1 scores, scoring parameter is set to 'f1'

    # calculating mean and standard deviation
    print("Mean: ", scores.mean())
    print("Standard Deviation: ", scores.std())

# ````````````````````End of Task-1````````````````````````````````````````````
