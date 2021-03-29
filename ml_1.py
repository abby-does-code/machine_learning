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

#print(digits.data[13])

#print(digits.data.shape)

print(digits.target[13])    #Row 13; target value is 3

print(digits.target.shape)  #This is the answer! 

#What is the target data one dimensional? The data has 64 features, but the target just has the answer!