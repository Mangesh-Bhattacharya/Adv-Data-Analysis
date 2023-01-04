import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ipaddress
import csv

# Read the data
dataset = pd.read_csv('lab8/NUSW-NB15_features.csv', sep=',', encoding='cp1252')
#import dataset Log_NB_1.csv
column = (dataset['Name'])
data = dataset.dropna(axis=1)
print(column)
print(data)