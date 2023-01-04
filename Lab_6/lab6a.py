import pandas
import numpy
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline

# Load the data
data = pandas.read_csv('data.csv')
# Add the column 'label' to hold the binary label (1 for 'M', 0 for 'B')
data['Label'] = data['Target'].map({'M': 1, 'B': 0})
# Create feature data frame without label information [drop the MD5, label and target columns]
X = data.drop(['MD5', 'Label', 'Target'], axis=1)
# Select the features using SelectKBest with chi2 and k=15
X_new = SelectKBest(chi2, k=15).fit_transform(X, data['Target'])
# List of features selected
features = X.columns[SelectKBest(chi2, k=15).fit(
    X, data['Target']).get_support()]
# Set the values for x_train and y_train
x_train = X_new
y_train = data['Target']
# Get the list of selected features from Lab 5
# List to store names of the methods used for feature selection
titles = []
# List to store the selected features
cases = []
# Perform Stratified K-Fold to split the dataset into 10 folds
#skfold = StratifiedKFold(n_splits=10).split(x_train, y_train)
# Use the following values for kernel:[linear, poly,rbf and sigmoid]
kernel = ['linear', 'poly', 'rbf', 'sigmoid']
# Use the following values for C:[100, 10, 1.0, 0.1, 0.001]
c_values = [100, 10, 1.0, 0.1, 0.001]
# Calculate and print the cross-validation score for each scenario
for k in kernel:
    for c in c_values:
        # Create a pipeline to send the data to the classifier
        SVC_classif = SVC(kernel=k, C=c)
        # Create a pipeline to send the data to the classifier and then to the feature selection method
        pipeline = make_pipeline(SelectKBest(chi2, k=15), SVC_classif)
        # Perform Stratified K-Fold to split the dataset into 10 folds
        skfold = StratifiedKFold(n_splits=10).split(x_train, y_train)
        # Calculate the mean of the cross validation scores
        scores = cross_val_score(
            pipeline, x_train, y_train, scoring='f1', cv=skfold)
        # Print the mean of the cross validation scores and the name of the feature selection method
        print("Mean: ", scores.mean())
        print("Kernel: ", k)
        print("C: ", c)
