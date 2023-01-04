import pandas
import numpy
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score, RandomizedSearchCV, GridSearchCV
from sklearn.utils.fixes import loguniform
from sklearn.metrics import f1_score

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

# Create SVC model object
SVC_classif = SVC()
# Define search space for hyperparameters
#search_space = [{'kernel': ['linear', 'poly', 'rbf', 'sigmoid'], 'C': loguniform(1e-3, 1e2)}]
search_space = dict()
search_space['kernel'] = ['linear', 'poly', 'rbf', 'sigmoid']
search_space['C'] = loguniform(1e-5, 100).rvs(10)
search_space['tol'] = loguniform(1e-5, 100).rvs(10)

# Define search for hyperparameters
search = GridSearchCV(SVC_classif, search_space,
                      scoring='f1', cv=10, n_jobs=500)
# Parameter tuning for C needs to be in list

# Perform search
search.fit(x_train, y_train)
# View best hyperparameters
#print('Best kernel:', search.best_estimator_.get_params()['kernel'])
#print('Best C:', search.best_estimator_.get_params()['C'])
#print('Best tol:', search.best_estimator_.get_params()['tol'])

# Print the best score and best parameter
print('Best score:', search.best_score_)
print('Best parameters:', search.best_params_)
print('f1 score:', f1_score(y_train, search.predict(x_train)))
