import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

warnings.filterwarnings("ignore")

df = pd.read_csv("insurance.csv")

print(df)


# EDA: Exploratory Data Analysis

print(df.shape)


print(df.head())

print(df.describe())

print(df.isnull().sum())

print(df.columns)

numeric_columns = ['age', 'bmi', 'children', 'charges']

for col in numeric_columns:
    plt.figure(figsize=(6, 4))
    sns.histplot(df[col], kde=True, bins=30) #kde=True for Kernel Density Estimation
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')
    # plt.show()
    
sns.countplot(x = df['children'])
plt.title('Count of Children')
plt.xlabel('Number of Children')
plt.ylabel('Count')
# plt.show()

sns.countplot(x = df['sex'])
plt.title('Count of Sex')
plt.xlabel('Sex')
plt.ylabel('Count')
# plt.show()

for col in numeric_columns:
    plt.figure(figsize=(6,4))
    sns.boxplot(x = df[col])
    # plt.show()
    
plt.figure(figsize=(8,6))
sns.heatmap(df.corr(numeric_only=True),annot=True)
# plt.show()


# Data Cleaning and Preprocessing

df_cleaned = df.copy()
print(df_cleaned.head())

print(df_cleaned.shape)
print(df_cleaned.drop_duplicates(inplace = True))

print(df_cleaned.shape)

print(df_cleaned.isnull().sum())

print(df_cleaned.dtypes)

print(df_cleaned['sex'].value_counts())

df_cleaned['sex'] = df_cleaned['sex'].map({'male': 0, 'female': 1})
print(df_cleaned.head())

print(df_cleaned['smoker'].value_counts())