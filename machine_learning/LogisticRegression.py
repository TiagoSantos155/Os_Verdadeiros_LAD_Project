import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.decomposition import PCA
import numpy as np
import os
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV_PATH = os.path.join(BASE_DIR, "../dataset/purchased_games_final.csv")

# Carregar os dados
df = pd.read_csv(DATASET_CSV_PATH)

# Usar apenas as primeiras 500000 linhas (ajuste conforme necessário)
df = df.head(500000)

# Selecionar apenas as colunas relevantes
cols = [
    'developers', 'genres', 'eur', 'release_date'
]
data = df[cols].copy()

# Definir o que é "caro" (acima da mediana)
price_threshold = data['eur'].median()
data['is_expensive'] = (data['eur'] > price_threshold).astype(int)

# Selecionar features relevantes
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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=69)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69)  # 0.5 x 0.3 = 0.15

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA diretamente aqui
n_components = 2
pca = PCA(n_components=n_components, svd_solver='auto', random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Treinar e avaliar o modelo Logistic Regression
fit_start = time.time()
logreg = LogisticRegression(max_iter=1000)
logreg.fit(X_train_pca, y_train)
fit_end = time.time()
train_time = fit_end - fit_start

# Avaliar no conjunto de validação para análise de threshold (opcional)
y_val_pred = logreg.predict(X_val_pca)
val_accuracy = accuracy_score(y_val, y_val_pred)

# Avaliar no conjunto de teste
y_pred = logreg.predict(X_test_pca)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Exibir gráfico de acurácia na validação e teste usando Tkinter
root = tk.Tk()
root.title("Acurácia Logistic Regression")

fig, ax = plt.subplots()
ax.bar(['Validação', 'Teste'], [val_accuracy, accuracy], color=['skyblue', 'orange'])
ax.set_ylim(0, 1)
ax.set_ylabel('Acurácia')
ax.set_title('Acurácia Logistic Regression')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def on_closing():
    root.destroy()
    print(f'Acurácia na validação: {val_accuracy:.2f}')
    print(f'Acurácia no conjunto de teste: {accuracy:.2f}')
    print(f'Precisão: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    print(f'Tempo de treino (fit): {train_time:.4f} segundos')

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
