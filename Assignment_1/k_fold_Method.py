from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, train_test_split, KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import plot_confusion_matrix

# Set the display properities
pd.options.display.width = 100
pd.options.display.precision = 2

# Get the firstline as header
cols = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', nrows=0).columns.tolist()

# Read the rest of the file
dataset = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', names=cols, skiprows=[0])
dataset = dataset.dropna(1)

# Split the data into train and test data
X=dataset.drop(['CreationYear', 'NumberOfSections', 'class'], axis=1)
y=dataset['class']

# Split the data test and train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create an instance of Pipeline
pipeline = make_pipeline(StandardScaler(), RandomForestClassifier(n_estimators=100, max_depth=4))

# Create an instance of K-Fold cross validation method
KFold = KFold(n_splits=10) # 10-fold cross validation is used here
kfold = KFold.split(X_train, y_train) # Get the indices of the training and test folds for each iteration
scores = [] # List to store the scores of each fold of cross validation

# Iterate over the folds and get the scores
for i, (train, test) in enumerate(kfold): # i is the fold number
    pipeline.fit(X_train.iloc[train, :], y_train.iloc[train]) # fit the model on the training data
    score = pipeline.score(X_train.iloc[test, :], y_train.iloc[test]) # get the score on the test data
    scores.append(score) # append the score to the list
    print('K-Fold: %2d, Train/Test Split: %s, Accuracy: %.3f' % (i+1, np.bincount(y_train.iloc[train]), score)) # print the score for each fold and split
print('\n\nCross-Validation accuracy: %.3f +/- %.3f' %(np.mean(scores), np.std(scores))) # print the mean and standard deviation of the scores for each fold

# Plot confusion matrix for the test data and the model trained on the entire training data
plot_confusion_matrix(pipeline, X_test, y_test, display_labels=['0', '1'], cmap=plt.cm.Blues, normalize='true')
plt.show()

# Scatter plot of the scores for each fold
plt.scatter(range(1, len(scores)+1), scores)
plt.xlabel('Number of iterations')
plt.ylabel('Accuracy')
plt.show()