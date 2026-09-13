# Logistic regression



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