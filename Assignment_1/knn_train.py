from matplotlib import pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import tree

df = pd.read_csv('A1_Mangesh/ClaMP_Raw-5184.csv')
df = df.dropna(1)
X = df.drop(['CreationYear', 'NumberOfSections', 'class'], axis=1)
y = df['class']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=0)

knn = KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski', metric_params=None, n_jobs=None)
clf = KNeighborsClassifier(p=1)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(y_pred)

# Plot the decision tree
plt.figure(figsize=(15, 7.5))
tree.plot_tree(clf.fit(X, y))
plt.show()