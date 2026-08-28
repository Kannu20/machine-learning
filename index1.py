import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings

from streamlit import columns
from sympy import per

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

df_cleaned['smoker'] = df_cleaned['smoker'].map({'no': 0, 'yes': 1})
print(df_cleaned)

# Rename columns
df_cleaned = df_cleaned.rename(
    columns={
        'sex': 'is_male',
        'smoker': 'is_smoker'
    }
)

# Check regions
print(df_cleaned['region'].value_counts())

# One-hot encoding region
df_cleaned = pd.get_dummies(
    df_cleaned,
    columns=['region'],
    drop_first=True
)

print(df_cleaned.head())
print(df_cleaned.dtypes)

# Feature Engineering and Extraction

sns.histplot(df['bmi'])
# plt.show()

df_cleaned['bmi_category'] = pd.cut(
    df_cleaned['bmi'],
    bins = [0, 18.5, 24.9, 29.9, float('inf')],
    labels = ['Underweight','Normal','Overweight','Obese']
    
)

print(df_cleaned)


df_cleaned = pd.get_dummies(df_cleaned, columns=['bmi_category'], drop_first=True)

df_cleaned = df_cleaned.astype(int)

print(df_cleaned.head())

print(df_cleaned.columns)

from sklearn.preprocessing import StandardScaler


cols = ['age','bmi','children']

scalar = StandardScaler()

df_cleaned[cols] = scalar.fit_transform(df_cleaned[cols])

print(df_cleaned.head())

from scipy.stats import pearsonr

selected_feature = [
    'age', 'is_male', 'bmi', 'children', 'is_smoker', 'region_northwest', 
    'region_southeast', 'region_southwest', 'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese'
]

correlations = {
    feature: pearsonr(df_cleaned[feature], df_cleaned['charges'])[0]
    for feature in selected_feature
}

correlations_df = pd.DataFrame(list(correlations.items()), columns=['Feature', 'Person Correlation'])
correlations_df = correlations_df.sort_values(
    by='Person Correlation',
    ascending=False
)

print(correlations_df)

cat_features = ['is_male', 'is_smoker', 'region_southwest', 'bmi_category_Normal', 'bmi_category_Overweight', 'bmi_category_Obese']

from scipy.stats import chi2_contingency
import pandas as pd

alpha = 0.05

df_cleaned['charges_bin'] = pd.qcut(
    df_cleaned['charges'],
    q=4,
    labels=False
)

chi2_results = {}

for col in cat_features:
    contingency_table = pd.crosstab(df_cleaned[col], df_cleaned['charges_bin'])
    chi2_stat, p_val, _, _ = chi2_contingency(contingency_table)
    decision = 'Reject Null (Keep Feature)' if p_val < alpha else 'Fail to Reject Null'
    chi2_results[col] = {
        'chi2_statistic' : chi2_stat,
        'p_value' : p_val,
        'Decision' : decision
    }
    
chi2_df = pd.DataFrame(chi2_results).T
chi2_df = chi2_df.sort_values(by='p_value', ascending=True)
print(chi2_df)

final_df = df_cleaned[['age', 'is_male', 'bmi', 'children', 'is_smoker','charges', 'region_southeast', 'bmi_category_Obese']]

print(final_df)