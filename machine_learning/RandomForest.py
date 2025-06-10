import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
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
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Treinar e avaliar o modelo Random Forest
fit_start = time.time()
rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
fit_end = time.time()
train_time = fit_end - fit_start

y_pred = rf.predict(X_test_scaled)
rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE no teste: {rmse:.2f}')
print(f'MAE no teste: {mae:.2f}')
print(f'R2 no teste: {r2:.2f}')
print(f'Tempo de treino (fit): {train_time:.4f} segundos')

# Exibir gráfico de importância das features usando Tkinter
root = tk.Tk()
root.title("Random Forest - Importância das Features")

fig, ax = plt.subplots()
importances = rf.feature_importances_
ax.bar(features, importances, color='forestgreen')
ax.set_ylabel('Importância')
ax.set_title('Importância das Features - Random Forest')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def on_closing():
    root.destroy()
    exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()