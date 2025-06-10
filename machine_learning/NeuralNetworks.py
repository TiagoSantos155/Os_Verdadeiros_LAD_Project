import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import PCA
import warnings
import numpy as np
import os
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt


# Ignorar o aviso de convergência do MLPRegressor
warnings.filterwarnings("ignore", category=ConvergenceWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV_PATH = os.path.join(BASE_DIR, "../dataset/purchased_games_final.csv")

# Carregar os dados
df = pd.read_csv(DATASET_CSV_PATH)
df = df.head(500000)

cols = ['developers', 'genres', 'eur', 'release_date']
data = df[cols].copy()

features = ['developers', 'genres', 'release_date']
X = data[features]
y = data['eur']

# Codificar variáveis categóricas
label_encoders = {}
for col in ['developers', 'genres']:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Converter release_date para ordinal
X.loc[:, 'release_date'] = pd.to_datetime(X['release_date'], errors='coerce').map(
    lambda x: x.toordinal() if pd.notnull(x) else 0
)

# Split
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=69)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=69)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA diretamente aqui (remover import e uso de apply_pca)
n_components = 2
pca = PCA(n_components=n_components, svd_solver='auto', random_state=69)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

results = []  # <-- Adicione esta linha antes de usar results

# Single layer (MLPRegressor com hidden_layer_sizes=(10,))
fit_start = time.time()
mlp_single = MLPRegressor(hidden_layer_sizes=(10,), max_iter=200, random_state=69)
mlp_single.fit(X_train_pca, y_train)
fit_end = time.time()
train_time_single = fit_end - fit_start

y_pred_single = mlp_single.predict(X_test_pca)
rmse_single = mean_squared_error(y_test, y_pred_single) ** 0.5
mae_single = mean_absolute_error(y_test, y_pred_single)
r2_single = r2_score(y_test, y_pred_single)

results.append(('Single Layer', rmse_single, mae_single, r2_single, train_time_single))

# Multi layer (MLPRegressor com hidden_layer_sizes=(50, 20))
fit_start = time.time()
mlp_multi = MLPRegressor(hidden_layer_sizes=(50, 20), max_iter=200, random_state=69)
mlp_multi.fit(X_train_pca, y_train)
fit_end = time.time()
train_time_multi = fit_end - fit_start

y_pred_multi = mlp_multi.predict(X_test_pca)
rmse_multi = mean_squared_error(y_test, y_pred_multi) ** 0.5
mae_multi = mean_absolute_error(y_test, y_pred_multi)
r2_multi = r2_score(y_test, y_pred_multi)

results.append(('Multi Layer', rmse_multi, mae_multi, r2_multi, train_time_multi))

# Mostrar resultados no terminal
for name, rmse, mae, r2, t in results:
    print(f"--- {name} ---")
    print(f"RMSE no teste: {rmse:.2f}")
    print(f"MAE no teste: {mae:.2f}")
    print(f"R2 no teste: {r2:.2f}")
    print(f"Tempo de treino (fit): {t:.4f} segundos")
    print()

# Gráfico comparativo usando Tkinter
root = tk.Tk()
root.title("Neural Network Regression Results")

fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(2)
rmse_vals = [rmse_single, rmse_multi]
mae_vals = [mae_single, mae_multi]
r2_vals = [r2_single, r2_multi]

ax.bar(index, rmse_vals, bar_width, label='RMSE')
ax.bar(index + bar_width, mae_vals, bar_width, label='MAE')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['Single Layer', 'Multi Layer'])
ax.set_ylabel('Erro')
ax.set_title('Comparação Neural Network Regressor')
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def on_closing():
    root.destroy()
    exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
