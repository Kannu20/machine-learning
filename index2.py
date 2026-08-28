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