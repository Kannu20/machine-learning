import numpy as np
from scipy.stats import norm

sample = [172, 174, 168, 169, 171, 173, 175, 170, 169, 172]
population_mean = 170
population_std = 3
samp_mean = np.mean(sample)
n = len(sample)
z_score = (samp_mean - population_mean) / (population_std / np.sqrt(n))
p_value = 2 * (1 - norm.cdf(abs(z_score)))
print(f"Z-Score: {z_score}")
print(f"P-Value: {p_value}")