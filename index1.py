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