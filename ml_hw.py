"""1.  Create a Seaborn pairplot graph (the book has an example in Unsupervised Machine Learning for the Iris Dataset Section 15.7.3) for the California Housing dataset. Try the Matplotlib features for panning and zooming the diagram. These are accessible via the icons in the Matplotlib window.

 

2. Re-implement the simple linear regression case study of Section 15.4 using the average yearly temperature data. How does the temperature trend compare to the average January high temperatures?"""

# Start#
from sklearn.datasets import fetch_california_housing

import seaborn as sns

california = fetch_california_housing()

# Create the dataframe

import pandas as pd

pd.set_option("max_columns", 9)  # This is the number of columns we need

pd.set_option("precision", 4)  # 4 digit float point

pd.set_option("display.width", None)  # Auto-detect the display

california_df = pd.DataFrame(california.data, columns=california.feature_nems)

print(california_df.describe())

"""
sns.set(font_scale=1.1)
sns.set_style('whitegrid')




#Code below uses pairplot to create a grid of graphs plotting each feature against each itself and the other specified features

grid = sns.pairplot(data=california_df, vars =california_df.columns[0:4], hue = '
"""
