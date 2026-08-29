import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings 

warnings.filterwarnings("ignore")

df = pd.read_csv("heart.csv")
print(df.head())

# EDA

print(df.columns)
print("df shape output ", df.shape)
print("df info output ", df.info)

print("df describe output ", df.describe())

print("df duplicated output ", df.duplicated().sum())

print(df['HeartDisease'].value_counts().plot(kind='bar', title='Count of Heart Disease'))

print("df isnull output ", df.isnull().sum())

def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)
plotting('Age', 1)
plotting('Cholesterol', 2)
plotting('MaxHR', 3)
plotting('RestingBP', 4)
plt.tight_layout()

plt.show()
print(plotting)
print(df['Cholesterol'].value_counts().plot(kind='bar', title='Count of Cholesterol'))

ch_mean = df.loc[df['Cholesterol'] !=0, 'Cholesterol'].mean()
print("Mean of Cholesterol excluding 0 values: ", ch_mean)

print(df['Cholesterol'].replace(0, ch_mean))
print(df['Cholesterol'].round(2))

restingbp_mean = df.loc[df['RestingBP'] !=0, 'RestingBP'].mean()
print("Mean of RestingBP excluding 0 values: ", restingbp_mean)
print(df['RestingBP'].replace(0, restingbp_mean))
print(df['RestingBP'].round(2))

def plotting(var, num):
    plt.subplot(2, 2, num)
    sns.histplot(df[var], kde=True)
plotting('Age', 1)
plotting('Cholesterol', 2)
plotting('MaxHR', 3)
plotting('RestingBP', 4)
plt.tight_layout()

plt.show()

sns.countplot(x = df['Sex'])
plt.title('Count of Sex')
plt.show()

sns.countplot(x = df['ChestPainType'],hue = df['HeartDisease'])
plt.title('Count of Chest Pain Type')
plt.show()

sns.countplot(x = df['Sex'],hue = df['HeartDisease'])
plt.title('Count of Sex with Heart Disease')
plt.show()

sns.boxplot(x = 'HeartDisease', y = 'Cholesterol', data = df)
plt.title('Boxplot of Cholesterol')
plt.show()

sns.violinplot(x = 'HeartDisease', y = 'Age', data = df)
plt.title('Violinplot of Age')
plt.show()

sns.heatmap(df.corr(numeric_only=True),annot=True)
plt.title('Correlation Heatmap')
plt.show()

# Data Preprocessing and Cleaning

df_encode = pd.get_dummies(df, drop_first=True)
print(df_encode.head())

print(df_encode.astype(int))

from sklearn.preprocessing import StandardScaler
numerical_cols = ['Age', 'RestingBP', 'Cholesterol', 'MaxHR', 'Oldpeak']
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
df_encode[numerical_cols] = scaler.fit_transform(df_encode[numerical_cols])
print(df_encode.head())