###########################################################################
# Name: Mangesh Bhattacharya
# Std No.: 039-251-145
# Course: SRT 521 - Adv. Data analysis
# Inst.: Dr. asma Paracha
# Date: 2022-11-25
# Description: Lab 9a
# Task 3: Multiple Linear Regression
# Guides: Aditi Singh and Asma Paracha
###########################################################################

# import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn import metrics

pd.options.display.width = 100
pd.options.display.precision = 2

# read in data
#cols = pd.read_csv('lab9/Lab9Data.csv',nrows=0).columns.tolist()
df = pd.read_csv('lab9/lab9Data_Cleaned.csv',skiprows=[0])
#print(df.head())

# Create a header list for the data frame and add it to the data frame
header = ['Date','Name','Country','BusinessType','BusinessSubType','BreachType','DataType','DataType2','InsideOutside','ThirdParty','ThirdPartyName','TotalAffected','RefPage','UID','StockSymbol','DataRecovered','ConsumerLawsuit','ArrestProsecution']
#print(header)
# Drop blank or empty columns
df = df.dropna(axis=1)

# Select your x1, x2 and y variables
x_var = df.iloc[:, 7].values.reshape(-1,1)
x2_var = df.iloc[:, 8].values.reshape(-1,1)
y_var = df.iloc[:, 2]

# Create the linear regression model. Call the fit method on your train data set
regressor = LinearRegression()
regressor.fit(x_var, x2_var, y_var)

 # Print the co-efficient of determination, root mean square error, intercept and slope.
print('Coefficient of determination: \n', regressor.score(x_var, x2_var, y_var))
print('Root mean square error: \n', np.sqrt(metrics.mean_squared_error(y_var,  regressor.predict(x2_var))))
print('Intercept: \n', regressor.intercept_)
print('Slope: \n', regressor.coef_)

# Call the predict method on the test and train data set
y_pred_train = regressor.predict(x_var)
y_pred_test = regressor.predict(x2_var)

# Compare the predicted values with the actual values and plot them on a line graph
plt.scatter(y_var, y_pred_train, color='red')
plt.scatter(x_var, y_pred_test, color='blue')
plt.plot(x2_var, y_pred_test, color='black')
plt.title('Actual vs Predicted')
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.legend(['Predicted', 'Actual'])
plt.show()

# Compare the results of two columns versus three and describe what is the optimal number of columns by printing the RMSE values
print('RMSE for 2 columns: \n', np.sqrt(metrics.mean_squared_error(y_var,  regressor.predict(x_var))))
print('RMSE for 3 columns: \n', np.sqrt(metrics.mean_squared_error(y_var,  regressor.predict(x2_var))))

# RMSE - Root Mean Square Error