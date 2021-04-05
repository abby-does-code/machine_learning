# Start#

# Missing the beginning

print(nyc.head(3))

print(nyc.Date.values)

print(nyc.Date.values.reshape(-1, 1))

print(nyc.Temperature.values)

from sklearn.model_selection import train_test_split

# When we want to do teh train_test_split, it has to be in a format Python is expecting
# so we don't get an error

# Estimator requires teh samples to be 2-dimensional while target can be one dimensional
X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)

# Fit method asks for X and y
# X_train: represents the data
# X_test: data; tests out the model
# Y_train: target
# Y_test: target

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)

"""OutPut: 
(93,1) (1 is the column; it's 2 dimensional)
(31,1)
(93,) (target is one dimensional)
(31)"""

# total data: 124
# % of data for train: 75%
# % of data for test: 25%

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

# fit method expects samples and targets for training
linear_regreassion.fit(X=X_train, y=y_train)  # Magic method that does the training
# Takes the data and identifies what target it gives; there's only one feature (year)

print(linear_regression.coef_)
# Coef = slope of the line
# Ouput = [0.01939167]
print(linear_regression.intercept_)
# Output = -0.307798...

predicted = linear_regression.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"Predicted: {p:.2f}, expected: {e:.2f}")
# Comparing every fifth method to see if they're the same or not

"""Output: 
last one: predicted: 36.94, expected: 39.70"""

# Not super accurate... not enough training data?
# More data = higher accuracy

predict = lambda x: linear_regression.coef_ * x + linear_regression.intercept_
# y = mx + b

print(predict(2021))
# Output: [38.88]
print(predict(1899))
# Output: [36.516]

import seaborn as sns

axes = sns.scatterplot(
    data=nyc,
    x="Date",
    y="Temperature",
    hue="Temperature",
    palette="winter",
    legend=False,
)

axes.set_ylim(10,70) #scale y-axis

import numpy as np 
x = np.array([min(nyc.Date.values), max(nyc.Date.values)]
print(x)

y = predict(x)
print(y)

import matplotlib.pyplot as plt

line = plt.plot(x,y)    #Feed the x and y arrays 
plt.show()