import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

x, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.60, random_state=42)

df = pd.DataFrame(x, columns=['Feature_1', 'Feature_2'])
print(df)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)

inertia = []
K_range = range(1, 11)

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)
    plt.show()
    print("Inertia Graph: ", inertia)
    
    
kmeans_final = KMeans(n_clusters=7, random_state=42)

cluster_labels = kmeans_final.fit_predict(X_scaled)

print(cluster_labels)

df['cluster'] = cluster_labels

sns.scatterplot(x=df['Feature_1'],
                y=df['Feature_2'],
                hue=df['cluster'],
                palette='viridis')
plt.show()