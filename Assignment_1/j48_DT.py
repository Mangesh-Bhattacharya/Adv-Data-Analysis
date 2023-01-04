# Import the needed libraries
from pandas import read_csv
from pandas import set_option
from pandas.core.groupby.generic import SeriesGroupBy
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.tree import export_text
import pandas as pd
import numpy as np
import graphviz
from matplotlib import pyplot

# Set the display properities
pd.options.display.width = 100
pd.options.display.precision = 2

# Get the firstline as header
cols = read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', nrows=0).columns.tolist()

# Read the rest of the file
dataset = read_csv('A1_Mangesh/ClaMP_Raw-5184.csv', names=cols, skiprows=[0])
dataset = dataset.dropna(1)
# Print to check if data is loaded correctly
# print(dataset)
# Our Target column is class, with 3 possible values Normal, suspicious and malicious
# Lets get a count on the number of samples belonging to each class to see how well the data is distributed

# DT works on number ONLY.
# We have 5184 rows and 10 cols in the dataset
# print(dataset.shape)

print("Total Samples: %d" % (len(dataset.index)))
print("Total Samples per Class")
gb = dataset.groupby('class')
print(gb.size())
malware = gb.get_group(0)
benignware = gb.get_group(1)
print('Percentage of Malware Group: %.2f %%' %
      (((len(malware.index)/len(dataset.index))*100)))
print('Percentage of Benignware Group: %.2f %%' %
      (((len(benignware.index)/len(dataset.index))*100)))

# Use The Decision Trees in SKLearn
# Split the data in x and y(the label)
X = dataset.drop(['CreationYear', 'NumberOfSections', 'class'], axis=1)
y = dataset['class']
#array = dataset.values
#X = array[:, 0:5184]
#y = array[:, 56:]

# Train model Decision Tree Classifier

# J48 use the split criteria= entropy which is for information gain
clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=3)

# Train the model
# works for number ONLY
clf = clf.fit(X, y)

# Print the model
tree.plot_tree(clf)
pyplot.show()

# Print to a pdf file
dot_data = tree.export_graphviz(clf, out_file=None,
                                filled=True, rounded=True,
                                special_characters=True)
graph = graphviz.Source(dot_data)
graph.render("A1_Mangesh/j48_DT")

# Load into a Text file
r = export_text(clf)
print(r)
