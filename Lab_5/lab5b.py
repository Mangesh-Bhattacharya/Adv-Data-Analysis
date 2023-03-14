import pandas
import numpy
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.feature_selection import SelectKBest, SelectPercentile, chi2, mutual_info_classif
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn import pipeline
import matplotlib.pyplot as plt

# Read the dataset from data.csv
cols = pandas.read_csv(
    'Semester-5\Data_Analysis Project\Lab_5\data.csv', nrows=0).columns.tolist()
# Read all row in the csv file and save the data in var data
dataset = pandas.read_csv(
    'Semester-5\Data_Analysis Project\Lab_5\data.csv', names=cols, skiprows=[0])
# Convert the data into numpy array
array = dataset.values
# Split the data into train and test data
X = array[:, 3:]
y = dataset['Target']

# List to store names of the methods used for feature selection
titles = []
# List to store the selected features
cases = []

# Split the data into 10 folds
skf = StratifiedKFold(n_splits=10, shuffle=True)
# Creates a pipeline with feature selection and model
pipe = pipeline.make_pipeline([('select', SelectKBest()), ('model', SVC())])

# List to store the accuracy of the models
accuracy = []

# List to store the names of the models
names = []

# Create a pipeline for SVM
svm = pipeline.Pipeline(
    [('feature_selection', SelectKBest(chi2, k=10)), ('classification', SVC())])
# Create a pipeline for Decision Tree
dt = pipeline.Pipeline([('feature_selection', SelectKBest(
    chi2, k=10)), ('classification', DecisionTreeClassifier())])
# Create a pipeline for Random Forest
rf = pipeline.Pipeline([('feature_selection', SelectKBest(
    chi2, k=10)), ('classification', RandomForestClassifier())])
# Create a pipeline for Naive Bayes
nb = pipeline.Pipeline([('feature_selection', SelectKBest(
    chi2, k=10)), ('classification', BernoulliNB())])

# Create a list of models
models = []
models.append(('SVM', svm))
models.append(('Decision Tree', dt))
models.append(('Random Forest', rf))
models.append(('Naive Bayes', nb))

# Evaluate each model in turn
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, shuffle=True)
    cv_results = cross_val_score(model, X, y, cv=kfold, scoring='f1')
    accuracy.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    print("Mean:" + str(cv_results.mean()))
    print("Standard Deviation:" + str(cv_results.std()))

# Create a boxplot to compare the accuracy of the models used for classification of the dataset (data.csv)
fig = plt.figure()
fig.suptitle('Comparison of Scores')
ax = fig.add_subplot(111)
plt.boxplot(accuracy)
ax.set_xticklabels(names)
plt.show()
