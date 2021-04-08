from sklearn.datasets import fetch_california_housing

california = fetch_california_housing()  # Bunch object

print(california.DESCR)

print(california.data.shape)  # Rows and columns
print(california.target.shape)  # Median house value for California districts

print(california.feature_names)  # Names of rows and columns

import pandas as pd

pd.set_option("precision", 4)  # 4 digit precision for floats
pd.set_option("max_columns", 9)  # display up to 9 columns in DF outputs
pd.set_option("display.width", None)  # auto-detect display

# Create initial DF using data in california.data
# Column names specified based on features of the sample

california_df = pd.DataFrame(california.data, columns=california.feature_names)

# Add a column to the DF for the median house vlues stored in california.target
california_df["MedHouseValue"] = pd.Series(california.target)

print(california_df.head())  # Peek at the first five rows

print(california.df.describe())  # Stats on the data; average, standard deviation, etc.

# Keyword argument frac specifies the fraction of the data to select (0.1 for 10%)
# and the ekyword argument random_state enables you to seed the random number generator.
# This allows you to reproduce teh same "Randomly" selected rows

import matplotlib.pyplot as pyplot
import seaborn as sns


sns.set_style("whitegrid")

for feature in california.feature_names:
    plt.figure(figsize=(8, 4.5))
    sns.scatterplot(
        data=sample_df, x=feature, y="MedHouseValue", palette="cool", legend=False
    )

plt.show()

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    california.data, california.target, random_state=11
)

print(X_test.shape)
print(X_train.shape)
print(y_train.shape)
print(y_test.shape)

from sklearn.linear_model import LinearRegression

linear_regression = LinearRegression()

linear_regression.fit(X=x_train, y=y_train)

# Using the enumerate method next!
##If the data contains categorical data, you'll have to do some pre-processing

# Enumerate allows you to get a counter and the value from the iterable at the same time!

for i, name in enumerate(california.feature_names):
    print(f"{name}:{linear_regression.coef_[i]}")

predicted = linear_regression.predict(x_test)
print(predicted[:5])

expected = y_test
print(expected[:5])

# CREATE A DF CONTAINING CLUMNS FOR THE EXPECTED VS PREDICTED VALUES:

df = pd.DataFrame()

df["Expected"] = pd.Series(expected)

df["Predicted"] = pd.SEries(predicted)

print(df[:10])
# Output should be two columns with expected and predicted

# Time to plot!

import matplotlib.pyplot as plt2

figure = plt2.figure(figsize=(9, 9))

axes = sns.scatterplot(
    data=df, x="Expected", y="Predicted", palette="cool", legend=False
)

# Set the x and y axes to use the same scale along both axes

start = min(expected.min(), predicted.min())
print(start)  # Giving us two values here; same scale for both axes

axes.setxlim(start, end)
axes.set_ylim(start, end)

line = plt2.plot([start, end], [start, end], "k--")
