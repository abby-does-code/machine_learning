# Start#

"""You are to apply skills you have acquired in Machine Learning to correctly predict the classification of a group of animals. The data has been divided into 3 files.

Classes.csv is a file describing the class an animal belongs to as well as the name of the class. The class number and class type are the two values that are of most importance to you.

animals_train.csv   download   - is the file you will use to train your model. There are 101 samples with 17 features. The last feature is the class number (corresponds to the class number from the classes file). This should be used as your target attribute. However, we want the target attribute to be the class type (Mammal, Bird, Reptile, etc.) instead of the class number (1,2,3,etc.).

animals_test.csv   download - is the file you will use to test your model to see if it can correctly predict the class that each sample belongs to. The first column in this file has the name of the animal (which is not in the training file).  Also, this file does not have a target attribute since the model should predict the target class.

Your program should produce a csv file that shows the name of the animal and their corresponding class as shown in this file -predictions.csv  """

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot

animal_class = pd.read_csv("animal_classes.csv")

animals_training = pd.read_csv("animals_train.csv")


print(animal_class.head(3))

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(
    animals_training.data, animals_training.target, random_state=11
)
# AttributeError: 'DataFrame' object has no attribute 'data'

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier()
knn.fit(X=data_train, y=target_train)

predicted = knn.predict(X=data_test)
predicted = [animal_class.target_names[i] for i in predicted]

expected = target_test
expected = [animal_class.target_names[i] for i in expected]
