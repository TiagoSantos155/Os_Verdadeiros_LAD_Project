import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import silhouette_score, accuracy_score, r2_score
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.svm import SVC
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV_PATH = os.path.join(BASE_DIR, "../dataset/purchased_games_final.csv")

# Carregar dados
df = pd.read_csv(DATASET_CSV_PATH)
df = df.head(50000)  # Amostra menor para visualização

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

# PCA para clustering
pca = PCA(n_components=2, random_state=69)
X_pca = pca.fit_transform(X_scaled)

# Para classificação/regressão
features = ['developers', 'genres']
X_class = data[features]
y_class = (data['eur'] > data['eur'].median()).astype(int)  # binário caro/barato

features_reg = ['developers', 'genres', 'release_date']
X_reg = data[features_reg]
y_reg = data['eur']

# Modelos a testar (apenas os que você tem)
model_builders = {
    #"KMeans": lambda: KMeans(n_clusters=4, random_state=69, n_init=10),
    #"DecisionTreeClassifier": lambda: DecisionTreeClassifier(max_depth=5, random_state=69),
    #"KNN": lambda: KNeighborsClassifier(n_neighbors=5),
    #"LinearRegression": lambda: LinearRegression(),
    #"LogisticRegression": lambda: LogisticRegression(max_iter=1000, random_state=69),
    #"RandomForest": lambda: RandomForestRegressor(n_estimators=10, random_state=69),
    #"Ridge": lambda: Ridge(alpha=1.0),
    #"Lasso": lambda: Lasso(alpha=0.1, max_iter=10000),
    "SVM": lambda: SVC(kernel='rbf', random_state=69),
    #"NaiveBayes": lambda: GaussianNB(),
    #"NeuralNetworks": lambda: MLPRegressor(hidden_layer_sizes=(10,), max_iter=200, random_state=69)
}

#Retirar o SVM demora muito FODASE

results = {}

# Cross-validation para cada modelo
total_models = len(model_builders)
model_names = list(model_builders.keys())
start_all = time.time()

for idx_model, (name, build_model) in enumerate(model_builders.items(), 1):
    scores = []
    print(f"\n[{idx_model}/{total_models}] Rodando cross-validation para: {name}")
    start_model = time.time()
    if name == "KMeans":
        # Clustering: silhouette score
        pseudo_labels = KMeans(n_clusters=4, random_state=69, n_init=10).fit_predict(X_pca)
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
        n_folds = kf.get_n_splits()
        for i, (train_idx, test_idx) in enumerate(kf.split(X_pca, pseudo_labels), 1):
            fold_start = time.time()
            X_test_fold = X_pca[test_idx]
            model = build_model()
            labels = model.fit_predict(X_test_fold)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            if n_clusters > 1 and n_clusters < len(X_test_fold):
                score = silhouette_score(X_test_fold, labels)
                scores.append(score)
            fold_end = time.time()
            elapsed = fold_end - start_model
            avg_per_fold = elapsed / i
            remaining = n_folds - i
            est_remaining = avg_per_fold * remaining
            print(f"  Fold {i}/{n_folds} | Tempo decorrido: {elapsed:.1f}s | Estimado restante: {est_remaining:.1f}s", end='\r')
        print()  # Nova linha após barra de progresso
        results[name] = np.mean(scores) if scores else None
    elif name in ["DecisionTreeClassifier", "KNN", "LogisticRegression", "NaiveBayes", "SVM"]:
        # Classificação: accuracy
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=69)
        n_folds = kf.get_n_splits()
        for i, (train_idx, test_idx) in enumerate(kf.split(X_class, y_class), 1):
            fold_start = time.time()
            X_train, X_test = X_class.iloc[train_idx], X_class.iloc[test_idx]
            y_train, y_test = y_class.iloc[train_idx], y_class.iloc[test_idx]
            scaler_c = StandardScaler()
            X_train_scaled = scaler_c.fit_transform(X_train)
            X_test_scaled = scaler_c.transform(X_test)
            model = build_model()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = accuracy_score(y_test, y_pred)
            scores.append(score)
            fold_end = time.time()
            elapsed = fold_end - start_model
            avg_per_fold = elapsed / i
            remaining = n_folds - i
            est_remaining = avg_per_fold * remaining
            print(f"  Fold {i}/{n_folds} | Tempo decorrido: {elapsed:.1f}s | Estimado restante: {est_remaining:.1f}s", end='\r')
        print()
        results[name] = np.mean(scores) if scores else None
    else:
        # Regressão: R2 score
        kf = KFold(n_splits=5, shuffle=True, random_state=69)
        n_folds = kf.get_n_splits()
        for i, (train_idx, test_idx) in enumerate(kf.split(X_reg), 1):
            fold_start = time.time()
            X_train, X_test = X_reg.iloc[train_idx], X_reg.iloc[test_idx]
            y_train, y_test = y_reg.iloc[train_idx], y_reg.iloc[test_idx]
            scaler_r = StandardScaler()
            X_train_scaled = scaler_r.fit_transform(X_train)
            X_test_scaled = scaler_r.transform(X_test)
            model = build_model()
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            score = r2_score(y_test, y_pred)
            scores.append(score)
            fold_end = time.time()
            elapsed = fold_end - start_model
            avg_per_fold = elapsed / i
            remaining = n_folds - i
            est_remaining = avg_per_fold * remaining
            print(f"  Fold {i}/{n_folds} | Tempo decorrido: {elapsed:.1f}s | Estimado restante: {est_remaining:.1f}s", end='\r')
        print()
        results[name] = np.mean(scores) if scores else None
    end_model = time.time()
    print(f"Modelo {name} finalizado em {end_model - start_model:.1f}s.")

end_all = time.time()
print(f"\nTempo total de execução: {end_all - start_all:.1f}s")

print("\nResultados cross-validation (média):")
for name, score in results.items():
    print(f"{name}: {score if score is not None else 'Não aplicável'}")
