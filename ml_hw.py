"""1.  Create a Seaborn pairplot graph (the book has an example in Unsupervised Machine Learning for the Iris Dataset Section 15.7.3) for the California Housing dataset. Try the Matplotlib features for panning and zooming the diagram. These are accessible via the icons in the Matplotlib window."""


"""2. Re-implement the simple linear regression case study of Section 15.4 using the average yearly temperature data. How does the temperature trend compare to the average January high temperatures?"""

# Start#
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt

import seaborn as sns

california = fetch_california_housing()

# Create the dataframe

import pandas as pd

pd.set_option("max_columns", 9)  # This is the number of columns we need

pd.set_option("precision", 4)  # 4 digit float point

pd.set_option("display.width", None)  # Auto-detect the display

california_df = pd.DataFrame(california.data, columns=california.feature_names)


# print(california_df.describe())

"""california_df["MedHouseValue"] = pd.Series(california.target)

## Created a data frame because it looks like that's what the graph is built off of below


sns.set(font_scale=1.1)
sns.set_style("whitegrid")


# Code below uses pairplot to create a grid of graphs plotting each feature against each itself and the other specified features

grid = sns.pairplot(
    data=california_df, vars=california_df.columns[0:4],palette="cool", hue="MedHouseValue", legend = False
)

OR
grid = sns.pairplot(data=california_df, vars=california_df.columns[0:4])"""


"""2. Re-implement the simple linear regression case study of Section 15.4 using the average yearly temperature data. How does the temperature trend compare to the average January high temperatures?"""

nyc = pd.read_csv("ave_hi_nyc_jan_1895-2018.csv")

nyc.columns = ["Date", "Temperature", "Anomaly"]

nyc.Date = nyc.Date.floordiv(100)

print(nyc.head(3))

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    nyc.Date.values.reshape(-1, 1), nyc.Temperature.values, random_state=11
)

print(X_train.shape)
print(X_test.shape)

from sklearn.linear_model import LinearRegression

linear_regr = LinearRegression()

linear_regr.fit(X=X_train, y=y_train)

print(linear_regr.coef_)


# Testing our model
##We use these predictions vs. expected lines to test how our model is doing
predicted = linear_regr.predict(X_test)

expected = y_test

for p, e in zip(predicted[::5], expected[::5]):
    print(f"predicted: {p:.2f},expected: {e:.2f}")

# Based on the output from this statement, our model SUCKS
##...either that or I'm doing something horribly wrong.

# NOW we're moving on to comparing two different data sets! How exciting.

predict = lambda x: linear_regr.coef_ * x + linear_regr.intercept_

print(predict(2019))
# This is alarming:
# OUTPUT : 3005.254

print(predict(1890))
# OUTPUT: 2813

# Seaborn imported above

axes = sns.scatterplot(
    data=nyc, x="Date", y="Temperature", palette="winter", legend=False
)

# data specifies where to pull the data
# hue specifies how colors are determined
# palette is Matplotlib color map
# legend specifies thatthe scatterplot shouldn't show one

# Scaling the range of values for better visualization

axes.set_ylim(10, 70)

# To create regression line, create array containing min and max date values from nyc.Date
##THESE MARK THE REGR LINE'S START AND END POINTS; VRY IMPORTANT!
import numpy as np

x = np.array([min(nyc.Date.values), max(nyc.Date.values)])

# Below, we pass the array to the predict lambda and produce another array with the corresponding values
# Corresponding values = y values

y = predict(x)

# Use Matplotlib to plot the line based on these two arrays

line = plt.plot(x, y)

plt.show()
