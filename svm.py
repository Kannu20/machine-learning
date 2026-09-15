#Support Vector Machine Implementation

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df = sns.load_dataset("titanic")

print(df.head())

print(df.columns)

print(df.info())

print(df.drop(['deck', 'embark_town', 'alive', 'class', 'who', 'adult_male', 'alone'], axis=1, inplace=True))

print(df.info())

print(df['age'].fillna(df['age'].mean(), inplace=True))

print(df.dropna(subset=['embarked'], inplace=True))

print(df.info())

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
print(df.head())
df['sex'] = le.fit_transform(df['sex'])
df['embarked'] = le.fit_transform(df['embarked'])

df = df.astype(int)
print(df.head())

x = df.drop(['survived'], axis=1)
y = df['survived']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression()

print(model.fit(x_train, y_train))

y_prod = model.predict(x_test)

print(y_prod)

print(y_test)

from sklearn.naive_bayes import GaussianNB

model_NB = GaussianNB()

model_NB.fit(x_train, y_train)

y_prod_NB = model_NB.predict(x_test)

print(y_prod_NB)

print(y_test)

accuracy = accuracy_score(y_test, y_prod_NB)
print("Accuracy score of Naive Bayes is " , accuracy)

print("confusion matix of naive bayes",confusion_matrix(y_test, y_prod_NB))

print("classifaction report of naive bayes is " , classification_report(y_test, y_prod_NB))

from sklearn.tree import DecisionTreeClassifier
model_DT = DecisionTreeClassifier(random_state=42)

model_DT.fit(x_train, y_train)

y_prod_DT = model_DT.predict(x_test)

print("Decision Tree result",y_prod_DT)

print("Decision Tree Confusion Matrix: ",confusion_matrix(y_test, y_prod_DT))

print("Decision Tress Classification Report", classification_report(y_test, y_prod_DT))

accuracy = accuracy_score(y_test, y_prod_DT)
print("Accuracy score of Decision Tree is " , accuracy)

from sklearn.svm import SVC

model_svm = SVC(kernel='rbf', random_state=42)

model_svm.fit(x_train, y_train)

model_svm_pred = model_svm.predict(x_test)

print("SVM result",model_svm_pred)

accuracy = accuracy_score(y_test, model_svm_pred)
print("Accuracy score of SVM is " , accuracy)

confusion = confusion_matrix(y_test, model_svm_pred)
print("SVM Confusion Matrix: ", confusion)

classification_rep = classification_report(y_test, model_svm_pred)
print("SVM Classification Report: ", classification_rep)