import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns 
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

x, y_true = make_blobs(n_samples=500, centers=3, cluster_std=0.5, random_state=42)

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

from sklearn.datasets import make_moons

x, y_true = make_moons(n_samples=500, noise=0.05, random_state=42)

from sklearn.cluster import KMeans, DBSCAN

df = pd.DataFrame(x, columns=['Feature_1', 'Feature_2'])

scaler = StandardScaler()
x_scaled = scaler.fit_transform(df)

kmeans = KMeans(n_clusters=2, random_state=42)
kmeans_labels = kmeans.fit_predict(x_scaled)

df['kmeans_cluster'] = kmeans_labels

sns.scatterplot(x=df['Feature_1'],
                y=df['Feature_2'],
                hue=df['kmeans_cluster'],
                palette='tab10')

plt.title('K-Means Clustering on Moons Dataset')
plt.show()