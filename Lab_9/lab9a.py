###########################################################################
# Name: Mangesh Bhattacharya
# Std No.: 039-251-145
# Course: SRT 521 - Adv. Data analysis
# Inst.: Dr. asma Paracha
# Date: 2022-11-25
# Description: Lab 9A
# Task 1: Understand the data
# Guides: Aditi Singh and Asma Paracha
###########################################################################
# import libraries to clean lab9Data.csv
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# set display width to 100 characters
pd.options.display.width = 100
# set display precision to 2 decimal places
pd.options.display.precision = 2

# read in data
#cols = pd.read_csv('lab9/Lab9Data.csv',nrows=0).columns.tolist()
df = pd.read_csv('lab9/features.csv',skiprows=[0])

# Create a header list for the data frame and add it to the data frame
header = ['Date','Name','Country','BusinessType','BusinessSubType','BreachType','DataType','DataType2','InsideOutside','ThirdParty','ThirdPartyName','TotalAffected','RefPage','UID','XREF1','StockSymbol','DataRecovered','ConsumerLawsuit','ArrestProsecution']
df.columns = header

# Remove the first 3 rows from the csv file
#df = df.drop(df.index[0:3])

# Drop blank or empty columns
df = df.dropna(axis=1)

# Functuion to clean data and return a list of cleaned data (To see the cleaned data, uncomment the print statement)
#def clean_data(df):
    # remove all non-numeric characters
#    df = df.replace(r'\D', '', regex=True)
    # remove all leading zeros
#    df = df.replace(r'^0', '', regex=True)
    # replace empty strings with NaN
#    df = df.replace('', np.nan)
    # convert column to numeric
#    df = df.apply(pd.to_numeric)
#    return df

# Print cleaned data
#print(clean_data(df))

# Use Label Encoder to convert NaN and blanks to numeric values for Lab9Data.csv
le = LabelEncoder()
df = df.apply(le.fit_transform)
print(df)

# Use train_test_split to split the data into training and testing sets
X = df[['UID','DataRecovered','ArrestProsecution']]
y = df['Country']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Print the shape of the training and testing sets
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

# Print the first 5 rows of the training and testing sets
print(X_train.head())
print(X_test.head())
print(y_train.head())
print(y_test.head())

# Print the number of rows in the training and testing sets
print(X_train.count())
print(X_test.count())
print(y_train.count())
print(y_test.count())

# Print the output of the data frame to a csv file called lab9DataCleaned.csv
#df.to_csv('lab9/lab9Data_Cleaned.csv', index=False)

# Plot the data frame to a bar graph and scatter plot for the following columns:
# TotalAffected, DataRecovered, ConsumerLawsuit, ArrestProsecution, Country, BusinessType, DataType, DataType2, InsideOutside, ThirdParty, ThirdPartyName, RefPage, StockSymbol
# Plot from lab9/lab9Data_Cleaned.csv
df.plot(kind='bar', x='Country', y='Name')
plt.show()

# Plot Scatter plot for TotalAffected, DataRecovered, ConsumerLawsuit, ArrestProsecution
df.plot(kind='line', x='DataRecovered', y='UID')
plt.show()

# Plot Scatter Plot for TotalAffected, DataRecovered, ConsumerLawsuit, ArrestProsecution
df.plot(kind='scatter', x='UID', y='BusinessType')
plt.show()