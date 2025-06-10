import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV_PATH = os.path.join(BASE_DIR, "../dataset/purchased_games_final.csv")

# Carregar os dados
df = pd.read_csv(DATASET_CSV_PATH)
df = df.head(5000)  # Usar apenas as primeiras 5000 linhas para comparação de kernels

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
# Primeiro split: 70% treino, 30% temporário
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
# Segundo split: 50% do temporário para validação e 50% para teste (0.5 x 0.3 = 0.15)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Treinar o modelo SVM e avaliar no conjunto de teste
fit_start = time.time()
svm = SVC(kernel='rbf', random_state=42)
svm.fit(X_train_scaled, y_train)
fit_end = time.time()
train_time = fit_end - fit_start

y_pred = svm.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

# Para datasets grandes, SVM pode ser muito lento, mas é obrigatório usar conforme o enunciado.
# Se precisar de testar kernels diferentes, altere o parâmetro kernel para 'linear', 'poly', ou 'rbf'.
# Exemplo para testar vários kernels:

kernels = ['linear', 'rbf', 'poly']
results = []

for kernel in kernels:
    fit_start = time.time()
    svm = SVC(kernel=kernel, random_state=42)
    svm.fit(X_train_scaled, y_train)
    fit_end = time.time()
    train_time = fit_end - fit_start

    y_pred = svm.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append((kernel, accuracy, precision, recall, f1, train_time))

# Mostrar resultados no terminal
print("Kernel\tAcurácia\tPrecisão\tRecall\tF1-score\tTempo de treino (s)")
for kernel, accuracy, precision, recall, f1, train_time in results:
    print(f"{kernel}\t{accuracy:.2f}\t\t{precision:.2f}\t\t{recall:.2f}\t{f1:.2f}\t\t{train_time:.4f}")

# Dizer qual kernel foi mais rápido
fastest = min(results, key=lambda x: x[5])
print(f"\nKernel mais rápido: {fastest[0]} ({fastest[5]:.4f} segundos)")

# Exibir gráfico de barras das métricas no Tkinter
root = tk.Tk()
root.title("Resultados SVM")

fig, ax = plt.subplots()
metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-score']
values = [accuracy, precision, recall, f1]
ax.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
ax.set_ylim(0, 1)
ax.set_ylabel('Valor')
ax.set_title('Métricas SVM (conjunto de teste)')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def on_closing():
    root.destroy()
    # Após fechar o Tkinter, imprime os resultados
    print(f'Acurácia no conjunto de teste: {accuracy:.2f}')
    print(f'Precisão: {precision:.2f}')
    print(f'Recall: {recall:.2f}')
    print(f'F1-score: {f1:.2f}')
    print(f'Tempo de treino (fit): {train_time:.4f} segundos')

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
