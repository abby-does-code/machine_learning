# START#
# Yay for machine learning! Subset of artificial intelligence!#

# Sklearn is a pre-developed machine learning algorithms
##estimator algorithms and pre-packaged datasets

from sklearn.datasets import load_digits

digits = load_digits()

# Two parts to machine learning: training and testing your module
##Supervised machine learning: you need to know what the target value is; data set is labelled and can easily classify what the data is
##Unsupervised machine learning has no label

# Each row in a data set is a sample
# Each sample has an associate label called a target
# Multi-classification: several different classes; and you nee dto predict which class the data falls into

# Regression models produce continuous output
##Linear regression can be simple or many; not as accurate
# Multiple linear regression can be a little more accurate but is mostly eh

# Data set, load the data, and explore (visualize) the data
##You might need to transform the data!

##Ten possible classes; tryign to get the program to recognize which class the number belongs in

# print(digits.DESCR)

# print(digits.data[13])

# print(digits.data.shape)

"""print(digits.target[13])  # Row 13; target value is 3

print(digits.target.shape)  # This is the answer!

# What is the target data one dimensional? The data has 64 features, but the target just has the answer!

# In order to fee dthe model a correct dimension, we have to fee dit the right array

# Let's look at the data we have!
import matplotlib.pyplot as plt

figure, axes = plt.subplots(nrows=4, ncols=6, figsize=(6, 4))
# python zip function bundles the 3 interables and produces one
for item in zip(axes.ravel(), digits.images, digits.target):
    axes, image, target = item
    # Displays multichannel or single-channel image data
    axes.imshow(image, cmap=plt.cm.gray_r)
    axes.set_xticks([])  # remove x-axis ticks
    axes.set_yticks([])  # remove y axis ticks
    axes.set_title(target)  # the target value of the image
plt.tight_layout()
plt.show()"""

# Time for the cool stuff
from sklearn.model_selection import (
    train_test_split,
)  # This will create a split into 4 variables (it's a tuple)

data_train, data_test, target_train, target_test = train_test_split(
    digits.data, digits.target, random_state=11
)
# We know what the target SHOULD be, but we want to compare what the actual result is
# Random state eleven  - when you dno't state the random_state, it pulls somethign weird from your computer.
###What is a random_state???###
print(data_train.shape)
print(target_train.shape)

print(data_test.shape)

print(target_test.shape)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
# Load the training data into the model using the fit method
# Note: KNeighbors Classifier fit method does not do calculations; just loads model
knn.fit(X=data_train, y=target_train)
# REturns an array contianing the predicted class of each test image: creates array of digits

predicted = knn.predict(X=data_test)
expected = target_test
print(predicted[:20])
print(expected[:20])
