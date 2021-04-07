# Start#

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as pyplot

animal_class = pd.read_csv("animal_classes.csv")

animals_train = pd.read_csv("animals_train.csv")


print(animal_class.head(3))