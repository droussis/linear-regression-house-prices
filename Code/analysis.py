import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline to be used in Jupyter Notebook


# Read the house data into a data frame
df = pd.read_csv('../Data_Set/kc_house_data.csv')

# Explore the house data
print(df.head())
print(df.describe())

# Drop the id and date columns as they are irrelevant to analysis
df = df.drop(['id', 'date'], axis=1)

# Explore the number and type of features
print('# of data:', len(df), 'and # of features:', len(df.columns))
# The number of features is small; it would be better to use normal equation

# Check types of data and first row
print(df.dtypes)
print(df.ix[0])

# Check the number of data points with NaN, if any
print('# of NaN:' len(df.isnull()))

# Specify the target variable and the features
target = df.iloc[:, 0].name
features = df.iloc[:, 1:].columns.tolist()
print(features)
print(df.corr().ix[:, 0])

# Remove features with correlation < 0.1 and update features
df = df.drop(['sqft_lot', 'condition', 'yr_built', 'zipcode', 'long',
             'sqft_lot15'], axis=1)
features = df.iloc[:, 1:].columns.tolist()

# Check updated features and store length
print(features)
len_of_features = len(features)

# Normalize the data
df = (df - df.mean())/df.std()

# Create X, y and theta
X = df.iloc[:, 1:]
ones = np.ones([len(df), 1])
X = np.concatenate((ones, X), axis=1)
y = df.iloc[:, 0:1].values
theta = np.zeros([1, len_of_features + 1])

# Check the size of the matrices
print(X.shape)
print(y.shape)
print(theta.shape)


# Compute Cost Function
def computecost(X, y, theta):
    H = X @ theta.T
    J = np.power((H - y), 2)
    sum = np.sum(J)/(2 * len(X))
    return sum


print(computecost(X, y, theta))
