import seaborn as sns 
import pandas as pd
from scipy.stats import f_oneway

df = sns.load_dataset('titanic')

print(df)

print(df[['age', 'pclass']].dropna())
print(df['pclass'].unique())


class_1 = df[df['pclass'] == 1]['age']
class_2 = df[df['pclass'] == 2]['age']
class_3 = df[df['pclass'] == 3]['age']
print(class_1)

f_stats, p_value = f_oneway(class_1,class_2,class_3)

print(f"F-Statistic: {f_stats}")

print(f"P-Value: {p_value}")


alpha = 0.05

if p_value < alpha:
    print("Reject the null hypothesis and there is a significant difference between in atleast one passenger class")
else:
    print("There is no significant difference")