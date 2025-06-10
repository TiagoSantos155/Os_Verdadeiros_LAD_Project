import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Normalizar (opcional para árvores, mas manter para consistência)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

results = []

# Decision Tree com profundidade 5
fit_start = time.time()
dtree_5 = DecisionTreeRegressor(max_depth=5, random_state=69)
dtree_5.fit(X_train_scaled, y_train)
fit_end = time.time()
train_time_5 = fit_end - fit_start

y_pred_5 = dtree_5.predict(X_test_scaled)
rmse_5 = mean_squared_error(y_test, y_pred_5) ** 0.5
mae_5 = mean_absolute_error(y_test, y_pred_5)
r2_5 = r2_score(y_test, y_pred_5)

results.append(('Depth 5', rmse_5, mae_5, r2_5, train_time_5))

# Decision Tree com profundidade 20
fit_start = time.time()
dtree_20 = DecisionTreeRegressor(max_depth=20, random_state=69)
dtree_20.fit(X_train_scaled, y_train)
fit_end = time.time()
train_time_20 = fit_end - fit_start

y_pred_20 = dtree_20.predict(X_test_scaled)
rmse_20 = mean_squared_error(y_test, y_pred_20) ** 0.5
mae_20 = mean_absolute_error(y_test, y_pred_20)
r2_20 = r2_score(y_test, y_pred_20)

results.append(('Depth 20', rmse_20, mae_20, r2_20, train_time_20))

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
root.title("Decision Tree Regression Results")

fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(2)
rmse_vals = [rmse_5, rmse_20]
mae_vals = [mae_5, mae_20]
r2_vals = [r2_5, r2_20]

ax.bar(index, rmse_vals, bar_width, label='RMSE')
ax.bar(index + bar_width, mae_vals, bar_width, label='MAE')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(['Depth 5', 'Depth 20'])
ax.set_ylabel('Erro')
ax.set_title('Comparação Decision Tree Regressor')
ax.legend()

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def on_closing():
    root.destroy()
    exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
