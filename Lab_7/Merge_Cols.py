# Convert Categorical string to numbers

import pandas as pd
import numpy as np

# Read the file
df = pd.read_csv(
    'Semester-5\Data_Analysis Project\Lab_7\Dataset_CS\Cyber Security Breaches.csv')

# Print the details about data
# print(dataset.dtypes)
# print(dataset.shape)

# Drop Columns
merge = df.drop([
                'Unnamed: 0'], axis='columns')

print(merge)

# Save changes to dataset
merge.to_csv('Semester-5\Data_Analysis Project\Lab_7\Dataset_CS\Cyber Security Breaches.csv',
             index=False)
