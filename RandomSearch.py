# Random Search
import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

print(df.head())

from sklearn.model_selection import train_test_split

x = df.drop('species', axis = 1)
y = df['species']

x_train , x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

from sklearn.svm import SVC

model_svm = SVC(C = 30, kernel='rbf',gamma = 'auto')

model_svm.fit(x_train, y_train)

print("SVM MODEL SCORE: ", model_svm.score(x_test, y_test))

from sklearn.model_selection import RandomizedSearchCV

classifier = RandomizedSearchCV((model_svm),{
    'C' : [1, 10, 20, 30],
    'kernel' : ['rbf', 'linear'],
},n_iter = 5, cv = 5, return_train_score = False)
    