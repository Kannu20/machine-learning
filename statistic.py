
import numpy as np
import matplotlib.pyplot as plt
arr = np.array([2,3,4,6,7,8,8,12,13,16,17,23,25,27,34,37,201])

print("Array:", arr)

Q1 = np.percentile(arr, 25)
Q3 = np.percentile(arr, 75)
print("Q1:", Q1)
print("Q3:", Q3)

IQR = Q3 - Q1
print("IQR:", IQR)

UF = Q3 + 1.5 * IQR # Upper Fence
LF = Q1 - 1.5 * IQR # Lower Fence
print("Upper Fence:", UF)
print("Lower Fence:", LF)

l = []
for i in arr:
    if i < LF or i > UF:
        l.append(i)

print("Outliers:", l)

import seaborn as sns

sns.boxplot(x = arr)

plt.show()