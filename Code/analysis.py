import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
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
print('# of NaN:', len(df.isnull()))

# Specify the target variable and the features
target = df.iloc[:, 0].name
features = df.iloc[:, 1:].columns.tolist()
print(features)

# Correlations of features with target variable
correlations = df.corr()
correlations['price']

# Remove features with correlation < 0.1
cor_target = abs(correlations['price'])
removed_features = cor_target[cor_target < 0.1]
print(removed_features)
df = df.drop(['sqft_lot', 'condition', 'yr_built', 'zipcode', 'long',
             'sqft_lot15'], axis=1)

# Use Pearson correlation matrix
fig_1 = plt.figure(figsize=(12, 10))
new_correlations = df.corr()
sns.heatmap(new_correlations, annot=True, cmap='Greens', annot_kws={'size': 8})
plt.title('Pearson Correlation Matrix')
plt.show()

# Remove highly intercorrelated features
highly_correlated_features = new_correlations[new_correlations > 0.75]
print(highly_correlated_features.fillna('-'))
df = df.drop(['sqft_above', 'sqft_living15'], axis=1)

# Update features and store their length
features = df.iloc[:, 1:].columns.tolist()
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

# Store target
target = y

# Check the size of the matrices
print("Dimensions of X:", X.shape)
print("Dimensions of y:", y.shape)
print("Dimensions of theta:", theta.shape)


# Define computecost function
def computecost(X, y, theta):
    H = X @ theta.T
    J = np.power((H - y), 2)
    sum = np.sum(J)/(2 * len(X))
    return sum


print(computecost(X, y, theta))

# Set iterations and alpha (learning rate)
alpha = 0.01
iterations = 500


# Define gradientdescent function
def gradientdescent(X, y, theta, iterations, alpha):
    cost = np.zeros(iterations)
    for i in range(iterations):
        H = X @ theta.T
        theta = theta - (alpha/len(X)) * np.sum(X * (H - y), axis=0)
        cost[i] = computecost(X, y, theta)
    return theta, cost


# Do Gradient Descent and print final theta
final_theta, cost = gradientdescent(X, y, theta, iterations, alpha)
print("Final theta is:", final_theta)

# Compute and print final cost
final_cost = computecost(X, y, final_theta)
print("Final cost is:", final_cost)

# Plot Iterations vs. Cost
fig_2, ax = plt.subplots(figsize=(10, 8))
ax.plot(np.arange(iterations), cost, 'r')
ax.set_xlabel('Iterations')
ax.set_ylabel('Cost')
ax.set_title('Iterations vs. Cost')
plt.show()


# Define rmse function
def rmse(target, final_theta):
    predictions = X @ final_theta.T
    return np.sqrt(((predictions[:, 0] - target[:, 0]) ** 2).mean())


# Compute and print Root Mean Squared Error
rmse_val = rmse(target, final_theta)
print("Root Mean Squared Error is:", rmse_val)
