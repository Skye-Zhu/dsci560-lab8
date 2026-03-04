import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

df = pd.read_csv("lab8_docs.tsv", sep="\t")

models = [
    ("d2v_50", "d2v_50_embeddings.npy"),
    ("d2v_100", "d2v_100_embeddings.npy"),
    ("d2v_200", "d2v_200_embeddings.npy"),
]

for name, file in models:

    print("\nEvaluating", name)

    X = np.load(file)

    # clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(X)

    score = silhouette_score(X, labels)

    print("Silhouette score:", score)

    # PCA visualization
    pca = PCA(n_components=2)
    X2 = pca.fit_transform(X)

    plt.figure(figsize=(6,5))
    plt.scatter(X2[:,0], X2[:,1], c=labels, s=8)
    plt.title(f"{name} clustering")
    plt.savefig(f"{name}_cluster.png", dpi=200)
    plt.close()