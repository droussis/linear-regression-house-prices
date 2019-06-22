import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Read the house data into a data frame
house_data = pd.read_csv('../Data_Set/kc_house_data.csv')
# Explore the house data
print(house_data.head())
y = house_data[['price', 'bedrooms', 'bathrooms', 'floors']]
print(y.head())
# Explore the number and type of features
print(len(house_data.columns))
print(house_data.ix[0])
print(house_data.dtypes)
x = house_data[house_data['bedrooms'] == 'NaN']
print(len(x))
# Drop the id and date columns
# house_data = house_data.drop(['id', 'date'])
