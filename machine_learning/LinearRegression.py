import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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

# Selecionar features e target para regressão
features = ['developers', 'genres', 'release_date']
X = data[features]
y = data['eur']

# Codificar variáveis categóricas ANTES do split
label_encoders = {}
for col in ['developers', 'genres']:
    le = LabelEncoder()
    X.loc[:, col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Converter release_date para ordinal (dias desde 1970-01-01)
X.loc[:, 'release_date'] = pd.to_datetime(X['release_date'], errors='coerce').map(
    lambda x: x.toordinal() if pd.notnull(x) else 0
)

# Separar em treino, validação e teste (70% treino, 15% validação, 15% teste)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)  # 0.5 x 0.3 = 0.15

# Normalizar os dados
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Aplicar PCA (redução de dimensionalidade)
n_components = 2
pca = PCA(n_components=n_components, svd_solver='auto', random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_val_pca = pca.transform(X_val_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Treinar e avaliar o modelo Linear Regression usando PCA
fit_start = time.time()
linreg = LinearRegression()
linreg.fit(X_train_pca, y_train)
fit_end = time.time()
train_time = fit_end - fit_start

# Avaliar no conjunto de validação
y_val_pred = linreg.predict(X_val_pca)
val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5  # Remover 'squared' para compatibilidade
val_mae = mean_absolute_error(y_val, y_val_pred)
val_r2 = r2_score(y_val, y_val_pred)

# Avaliar no conjunto de teste
y_pred = linreg.predict(X_test_pca)
rmse = mean_squared_error(y_test, y_pred) ** 0.5  # Remover 'squared' para compatibilidade
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Avaliar acurácia (R2) nos três conjuntos
train_pred = linreg.predict(X_train_pca)
train_r2 = r2_score(y_train, train_pred)
val_r2 = r2_score(y_val, y_val_pred)
test_r2 = r2_score(y_test, y_pred)

# R2 é a acurácia para modelos de regressão.
# Ele mede o quanto o modelo explica a variação dos dados (quanto mais próximo de 1, melhor).
# Para classificação usamos "accuracy", para regressão usamos "R2" como métrica principal de "acurácia".

# Exibir gráfico de RMSE na validação e teste usando Tkinter
root = tk.Tk()
root.title("RMSE Linear Regression")

fig, ax = plt.subplots()
ax.bar(['Validação', 'Teste'], [val_rmse, rmse], color=['skyblue', 'orange'])
ax.set_ylabel('RMSE')
ax.set_title('RMSE Linear Regression')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

# Após avaliar o modelo, converta as previsões para 0 ou 1 com base em um threshold (ex: média ou mediana dos valores previstos)
# Aqui, vamos usar a mediana dos valores previstos como threshold para classificação binária
threshold = np.median(y_pred)
y_pred_class = (y_pred >= threshold).astype(int)
y_test_class = (y_test >= threshold).astype(int)

# Calcule a acurácia da classificação binária
accuracy = (y_pred_class == y_test_class).mean()

# Imprimir os resultados diretamente no terminal (antes do Tkinter)
print(f'RMSE na validação: {val_rmse:.2f}')
print(f'RMSE no conjunto de teste: {rmse:.2f}')
print(f'MAE no conjunto de teste: {mae:.2f}')
print(f'R2 no treino: {train_r2:.2f}')
print(f'R2 na validação: {val_r2:.2f}')
print(f'R2 no teste: {test_r2:.2f}')
print(f'Acurácia binária (0/1) no teste: {accuracy:.2f}')
print(f'Tempo de treino (fit): {train_time:.4f} segundos')

def on_closing():
    root.destroy()
    exit(0)  # Termina o programa após mostrar os dados

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()
