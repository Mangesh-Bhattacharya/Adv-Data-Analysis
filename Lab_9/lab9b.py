# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics

# set display width to 100 characters
pd.options.display.width = 100
# set display precision to 2 decimal places
pd.options.display.precision = 2 

# read in data
#cols = pd.read_csv('lab9/Lab9Data.csv',nrows=0).columns.tolist()
df = pd.read_csv('lab9/lab9Data_Cleaned.csv',skiprows=[0])
#print(df.head())

# Create a header list for the data frame and add it to the data frame
header = ['Date','Name','Country','BusinessType','BusinessSubType','BreachType','DataType','DataType2','InsideOutside','ThirdParty','ThirdPartyName','TotalAffected','RefPage','UID','StockSymbol','DataRecovered','ConsumerLawsuit','ArrestProsecution']
#print(header)

# independent variables
X = df.iloc[:, :8]
# dependent variable
y = df.iloc[:, 2]
print(X)
print(y)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

#  Create the linear regression model. Call the fit method on train data set
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#  Print the intercept and coefficients of the model
print('Intercept: \n', regressor.intercept_)
print('Coefficients: \n', regressor.coef_)
print('R^2: \n', regressor.score(X, y))

# Call the predict method on the test and train data set
y_pred_train = regressor.predict(X_train)
y_pred_test = regressor.predict(X_test)

# Compare the predicted values with the actual values and plot them on a line graph
plt.scatter(y_train, y_pred_train, color='red')
plt.scatter(y_test, y_pred_test, color='blue')
plt.plot(y_test, y_pred_test, color='black')
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend(['Predicted', 'Actual'])
plt.show()

# Describe how accurately the model works with the new data set
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred_test))
# The mean squared error is a measure of the quality of an estimator
print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred_test))
# RMSE is a measure of how spread out these residuals are (i.e. how much the model is predicting incorrectly)
print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, y_pred_test)))
