import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import os
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV_PATH = os.path.join(BASE_DIR, "../dataset/purchased_games_final.csv")

# Carregar os dados
df = pd.read_csv(DATASET_CSV_PATH)
df = df.head(5000)

# Remover outliers do target (eur) usando IQR
Q1 = df['eur'].quantile(0.25)
Q3 = df['eur'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['eur'] >= Q1 - 1.5 * IQR) & (df['eur'] <= Q3 + 1.5 * IQR)]

# Features e target
features = ['developers', 'genres', 'release_date']
X = df[features].copy()
y = df['eur']

# Codificar variáveis categóricas
for col in ['developers', 'genres']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))

# Converter release_date para ordinal
X['release_date'] = pd.to_datetime(X['release_date'], errors='coerce').map(
    lambda x: x.toordinal() if pd.notnull(x) else 0
)

# Split simples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=69)

# Normalizar
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Testar diferentes valores de max_depth
import time
rmse_list = []
mae_list = []
r2_list = []
train_times = []
depths = list(range(1, 50))

for d in depths:
    start = time.time()
    rf = RandomForestRegressor(n_estimators=25, max_depth=d, random_state=69, n_jobs=-1)
    rf.fit(X_train_scaled, y_train)
    end = time.time()
    train_time = end - start
    y_pred = rf.predict(X_test_scaled)
    y_pred = np.where(y_pred < 1, 0, y_pred)
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    rmse_list.append(rmse)
    mae_list.append(mae)
    r2_list.append(r2)
    train_times.append(train_time)
    print(f"max_depth={d}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}, Tempo treino={train_time:.4f}s")

# Mostrar os 3 melhores valores (menor RMSE)
results = list(zip(depths, rmse_list, mae_list, r2_list, train_times))
results_sorted = sorted(results, key=lambda x: x[1])  # Ordena por RMSE
print("\nTop 3 melhores valores (menor RMSE):")
for i in range(3):
    d, rmse, mae, r2, t = results_sorted[i]
    print(f"max_depth={d}: RMSE={rmse:.2f}, MAE={mae:.2f}, R2={r2:.2f}, Tempo treino={t:.4f}s")

# Plot dos resultados
plt.figure(figsize=(12, 6))
plt.plot(depths, rmse_list, marker='o', label='RMSE')
plt.plot(depths, mae_list, marker='o', label='MAE')
plt.plot(depths, r2_list, marker='o', label='R2')
plt.xlabel('Profundidade Máxima (max_depth)')
plt.title('Desempenho do Random Forest vs Profundidade Máxima (25 árvores)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))
plt.plot(depths, train_times, marker='o', color='orange', label='Tempo de treino (s)')
plt.xlabel('Profundidade Máxima (max_depth)')
plt.ylabel('Tempo de treino (s)')
plt.title('Tempo de treino vs Profundidade Máxima')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
