# Start#

"""You are to apply skills you have acquired in Machine Learning to correctly predict the classification of a group of animals. The data has been divided into 3 files.

Classes.csv is a file describing the class an animal belongs to as well as the name of the class. The class number and class type are the two values that are of most importance to you.

animals_train.csv   download   - is the file you will use to train your model. There are 101 samples with 17 features. The last feature is the class number (corresponds to the class number from the classes file). This should be used as your target attribute. However, we want the target attribute to be the class type (Mammal, Bird, Reptile, etc.) instead of the class number (1,2,3,etc.).

animals_test.csv   download - is the file you will use to test your model to see if it can correctly predict the class that each sample belongs to. The first column in this file has the name of the animal (which is not in the training file).  Also, this file does not have a target attribute since the model should predict the target class.

Your program should produce a csv file that shows the name of the animal and their corresponding class as shown in this file -predictions.csv  """

import pandas as pd


animal_class = pd.read_csv("animal_classes.csv")

X = pd.read_csv("animals_train.csv")

X.columns = [
    "hair",
    "feathers",
    "eggs",
    "milk",
    "airborne",
    "aquatic",
    "predator",
    "toothed",
    "backbone",
    "breathes",
    "venomous",
    "fins",
    "legs",
    "tail",
    "domestic",
    "catsize",
    "class_number",
]

y = X["class_number"]
X = X.drop(columns="class_number")


from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X, y)


test = pd.read_csv("animals_test.csv")
test_data = test.drop(columns="animal_name")  # feed everything except the target

predicted = knn.predict(X=test_data)

print(predicted[:10])

# predicted = [animal_class.target_names[i] for i in predicted]

class_number = animal_class["Class_Number"]

animal_name = test["animal_name"]

class_names = animal_class["Class_Type"]

print(class_names[:10])

# predicted = [class_number[i] for i in predicted]
# print(predicted)
name_num_dict = {
    "1": "Mammal",
    "2": "Bird",
    "3": "Reptile",
    "4": "Fish",
    "5": "Amphibian",
    "6": "Bug",
    "7": "Invertebrate",
}

for i in predicted:
    predicted[i] = 