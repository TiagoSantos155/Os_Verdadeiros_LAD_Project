import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV_PATH = os.path.join(BASE_DIR, "../dataset/purchased_games_final.csv")

# Carregar os dados
df = pd.read_csv(DATASET_CSV_PATH)

# Usar apenas as primeiras 500000 linhas
df = df.head(500000)

# Selecionar apenas as colunas relevantes
cols = [
    'developers', 'genres', 'eur', 'release_date'
]
data = df[cols].copy()

# Definir o que é "caro" (acima da mediana)
# Cria a coluna 'is_expensive' com base na mediana do preço ('eur')
price_threshold = data['eur'].median()
data['is_expensive'] = (data['eur'] > price_threshold).astype(int)

# Selecionar features relevantes (remover country)
features = ['developers', 'genres']
X = data[features]
y = data['is_expensive']

# Codificar variáveis categóricas ANTES do split
label_encoders = {}
for col in features:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Separar em treino, validação e teste (70% treino, 15% validação, 15% teste)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 0.5 x 0.3 = 0.15

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Encontrar o número ótimo de vizinhos (k) usando validação cruzada no conjunto de treino
k_range = range(1, 21)
cv_scores = []
fit_times = []
start_time = time.time()
total_k = len(k_range)
for idx, k in enumerate(k_range, 1):
    knn = KNeighborsClassifier(n_neighbors=k)
    fit_start = time.time()
    knn.fit(X_train_scaled, y_train)
    fit_end = time.time()
    fit_times.append(fit_end - fit_start)
    score = knn.score(X_val_scaled, y_val)
    cv_scores.append(score)
    elapsed = time.time() - start_time
    avg_time_per_iter = elapsed / idx
    remaining = total_k - idx
    est_remaining = avg_time_per_iter * remaining
    print(f'Progresso: {idx}/{total_k} | Tempo decorrido: {elapsed:.1f}s | Estimado restante: {est_remaining:.1f}s', end='\r')

print()  # Para nova linha após barra de progresso

# Selecionar o melhor k ANTES do Tkinter
optimal_k = k_range[np.argmax(cv_scores)]

# Treinar o modelo final e avaliar no conjunto de teste ANTES do Tkinter
knn = KNeighborsClassifier(n_neighbors=optimal_k)
fit_start = time.time()
knn.fit(X_train_scaled, y_train)
fit_end = time.time()
train_time = fit_end - fit_start

y_pred = knn.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Exibir o gráfico usando Tkinter
root = tk.Tk()
root.title("Acurácia na validação para cada valor de k")

fig, ax = plt.subplots()
ax.plot(k_range, cv_scores)
ax.set_xlabel('Número de vizinhos (k)')
ax.set_ylabel('Acurácia na validação')
ax.set_title('Escolha do número ótimo de vizinhos para K-NN')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def on_closing():
    root.destroy()
    # Após fechar o Tkinter, imprime os resultados
    print(f'Número ótimo de vizinhos: {optimal_k}')
    print(f'Acurácia no conjunto de teste: {accuracy:.2f}')
    print(f'Precisão: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    print(f'Tempo de treino (fit): {train_time:.4f} segundos')

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()