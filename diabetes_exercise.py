""" Using the Diabetes dataset that is in scikit-learn, answer the questions below and create a scatterplot
graph with a regression line """

import matplotlib.pylab as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets


# how many samples and How many features?
from sklearn.datasets import load_diabetes

diabetes = load_diabetes()

print(diabetes.data.shape)

##Samples = 442
##Features = 10

# What does feature s6 represent?

## Numpy arary?????

# print out the coefficient
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    diabetes.data, diabetes.target, random_state=11
)

print(X_train.shape)
print(X_test)
print(y_train.shape)
print(y_test.shape)


# print out the intercept
from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()
linear_regression.fit(X=X_train, y=y_train)

print(linear_regression.coef_)
print(linear_regression.intercept_)

predicted = linear_regression(X_test)
expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"Predicted: {p:.2f}, expected: {e:.2f}")


# create a scatterplot with regression line
 x = np.linspace(0,30,100)
 y = x

 plt.plot(x,y)
 plt.show()