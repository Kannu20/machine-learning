import numpy as np
import pandas as pd
import seaborn as sns 
import matplotlib.pyplot as plt

df = sns.load_dataset('iris')

print(df.head())

print(df['species'].unique())

from sklearn.model_selection import train_test_split

x = df.drop('species', axis = 1)
y = df['species']

x_train , x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.33, random_state=42
)

from sklearn.neighbors import KNeighborsClassifier

# model_knn = KNeighborsClassifier(n_neighbors=12) # In this situation model give 100% accuracy score so it is the condition of overfit
model_knn = KNeighborsClassifier(n_neighbors=2) 

model_knn.fit(x_train, y_train)

print("KNN MODEL SCORE: ",model_knn.score(x_test, y_test))

from sklearn.svm import SVC

model_svm = SVC(C = 30, kernel='rbf',gamma = 'auto')

model_svm.fit(x_train, y_train)

print("SVM MODEL SCORE: ", model_svm.score(x_test, y_test))

# now let's use the gird search cv


from sklearn.model_selection import GridSearchCV

classifier = GridSearchCV((model_svm),{
    'C' : [1, 10, 20, 30],
    'kernel' : ['rbf']
}, cv = 5, return_train_score = False)

classifier.fit(x, y)

print("classifier result: ",classifier.cv_results_)

results = pd.DataFrame(classifier.cv_results_)

print(results)

print(results[['param_C', 'mean_test_score', 'param_kernel']])
