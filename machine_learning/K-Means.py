import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV_PATH = os.path.join(BASE_DIR, "../dataset/purchased_games_final.csv")

# Carregar dados
df = pd.read_csv(DATASET_CSV_PATH)
df = df.head(25)  # Amostra menor para visualização

# Selecionar features relevantes
cols = ['developers', 'genres', 'eur', 'release_date']
data = df[cols].copy()

# Codificar variáveis categóricas
for col in ['developers', 'genres']:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col].astype(str))

# Converter release_date para ordinal
data['release_date'] = pd.to_datetime(data['release_date'], errors='coerce').map(
    lambda x: x.toordinal() if pd.notnull(x) else 0
)

# Normalizar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data)

# Reduzir para 2D para visualização
pca = PCA(n_components=2, random_state=69)
X_pca = pca.fit_transform(X_scaled)

# Encontrar número ótimo de clusters (Elbow e Silhouette)
inertia = []
silhouette = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=69, n_init=10)
    labels = kmeans.fit_predict(X_pca)
    inertia.append(kmeans.inertia_)
    silhouette.append(silhouette_score(X_pca, labels))

optimal_k = K_range[np.argmax(silhouette)]

# KMeans com número ótimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=69, n_init=10)
labels_kmeans = kmeans.fit_predict(X_pca)

# Clustering hierárquico (ward linkage)
Z = linkage(X_pca, method='ward')
agg = AgglomerativeClustering(n_clusters=optimal_k, linkage='ward')
labels_agg = agg.fit_predict(X_pca)

# Exibir gráficos em Tkinter
root = tk.Tk()
root.title("Clustering Analysis (KMeans & Hierarchical)")

fig, axs = plt.subplots(2, 2, figsize=(12, 10))

# Elbow
axs[0, 0].plot(K_range, inertia, marker='o')
axs[0, 0].set_title('Elbow Method (Inertia)')
axs[0, 0].set_xlabel('Número de clusters')
axs[0, 0].set_ylabel('Inertia')

# Silhouette
axs[0, 1].plot(K_range, silhouette, marker='o', color='orange')
axs[0, 1].set_title('Silhouette Score')
axs[0, 1].set_xlabel('Número de clusters')
axs[0, 1].set_ylabel('Silhouette')

# KMeans scatter
scatter = axs[1, 0].scatter(X_pca[:, 0], X_pca[:, 1], c=labels_kmeans, cmap='tab10', s=10)
axs[1, 0].set_title(f'KMeans Clusters (k={optimal_k})')
axs[1, 0].set_xlabel('PC1')
axs[1, 0].set_ylabel('PC2')

# Dendrograma (menos profundidade, sem nomes das amostras)
dendrogram(
    Z,
    truncate_mode='level',
    p=5,  # Reduzido para mostrar menos níveis de profundidade
    ax=axs[1, 1],
    color_threshold=None,
    no_labels=True
)
axs[1, 1].set_title('Hierarchical Clustering Dendrogram')
axs[1, 1].set_xlabel('Amostras')
axs[1, 1].set_ylabel('Distância')

fig.tight_layout()
canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def on_closing():
    root.destroy()
    print(f"Número ótimo de clusters (silhouette): {optimal_k}")

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
