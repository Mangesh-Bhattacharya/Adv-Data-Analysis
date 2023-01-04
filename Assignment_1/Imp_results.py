import graphviz
from matplotlib import pyplot as plt
import pandas as pd
from sklearn import tree
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

# Set the display properities
pd.options.display.width = 100
pd.options.display.precision = 2

# Get the firstline as header
cols = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', nrows=0).columns.tolist()

# Read the rest of the file
dataset = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', names=cols, skiprows=[0])
dataset = dataset.fillna(0)

# Split the data in x and y
X = dataset.drop(['CreationYear', 'NumberOfSections', 'class'], axis=1)
y = dataset['class']

# Split the data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

## Define the model
model=SVC()

## Defining the hyperparameter to be tunned
param_grid = {'C': [0.1, 1, 10, 100, 1000], 'kernel': ['rbf']}
## Kernel :choice of kernel that will control the manner in which the input variables will be projected
test_kernel=['poly', 'rbf','linear']

## GridSearchCV
grid = GridSearchCV(model, param_grid, refit=True, verbose=3)

## Fit the model
grid.fit(X_train, y_train)

## Print the best parameters
print(grid.best_params_)
# Print the best estimator
print(grid.best_estimator_)
# Print the best score
print(grid.best_score_)
# Print the best index
print(grid.best_index_)

## Predict the model
y_pred = grid.predict(X_test)

## Print the accuracy
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# Plot the confusion matrix
disp = plot_confusion_matrix(grid, X_test, y_test, cmap=plt.cm.Blues, normalize='true')
disp.ax_.set_title('Confusion Matrix')
print(disp.confusion_matrix)
plt.show()

# Scatter plot of the data points (Has issues loading)
#dataset.plot(kind='scatter', x='NumberOfSections', y='CreationYear', c='class', cmap= 'nipy_spectral')
#plt.title('Scatter plot of Accuracy')
#plt.show()
