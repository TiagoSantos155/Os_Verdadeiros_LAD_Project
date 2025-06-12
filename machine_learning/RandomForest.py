import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.tree import plot_tree
import numpy as np
import os
import time
import tkinter as tk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
import tkinter.ttk as ttk
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_CSV_PATH = os.path.join(BASE_DIR, "../dataset/purchased_games_final.csv")

# Carregar os dados
df = pd.read_csv(DATASET_CSV_PATH)
df = df.head(11258717)

# Remover outliers do target (eur) usando IQR
Q1 = df['eur'].quantile(0.25)
Q3 = df['eur'].quantile(0.75)
IQR = Q3 - Q1
df = df[(df['eur'] >= Q1 - 1.5 * IQR) & (df['eur'] <= Q3 + 1.5 * IQR)]

# Nota: Valores 0 em 'eur' podem indicar jogos gratuitos, mas também podem ser dados faltantes ou erros no dataset.
# Aqui, consideramos 'eur == 0' como proxy para "grátis", mas isso pode não ser perfeito.
# Se houver um campo mais confiável para indicar jogos gratuitos, use-o como target.

# Usar as colunas 'developers', 'genres' e 'release_date' como features (removido 'total_achievements')
cols = ['developers', 'genres', 'release_date']
data = df[cols].copy()

features = ['developers', 'genres', 'release_date']
X = df[features].copy()
y = df['eur']

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

# Remover PCA: usar os dados normalizados diretamente

# Treinar e avaliar o modelo Random Forest Regressor
fit_start = time.time()
rf = RandomForestRegressor(n_estimators=25, max_depth=43, random_state=69, n_jobs=-1)
rf.fit(X_train_scaled, y_train)
fit_end = time.time()
train_time = fit_end - fit_start

y_pred = rf.predict(X_test_scaled)
# Ajuste: valores previstos abaixo de 0.5 são assumidos como 0
y_pred = np.where(y_pred < 1, 0, y_pred)

rmse = mean_squared_error(y_test, y_pred) ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'RMSE no teste: {rmse:.2f}')
print(f'MAE no teste: {mae:.2f}')
print(f'R2 no teste: {r2:.2f}')
print(f'Tempo de treino (fit): {train_time:.4f} segundos')

# Guardar o modelo treinado como .pkl na pasta Modelos
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../Modelos/random_forest_model.pkl")
joblib.dump(rf, MODEL_SAVE_PATH)
print(f"Modelo Random Forest guardado em: {MODEL_SAVE_PATH}")

# Guardar os encoders e scaler usados no treino
ENCODERS_SAVE_PATH = os.path.join(BASE_DIR, "../Modelos/random_forest_encoders.pkl")
encoders_scaler = {
    "label_encoders": label_encoders,
    "scaler": scaler,
    "features": features
}
joblib.dump(encoders_scaler, ENCODERS_SAVE_PATH)
print(f"Encoders e scaler guardados em: {ENCODERS_SAVE_PATH}")

# Exibir gráfico de importância das features originais usando Tkinter
root = tk.Tk()
root.title("Random Forest - Importância das Features Originais")

fig, ax = plt.subplots()
rf_orig = RandomForestRegressor(n_estimators=25, max_depth=43, random_state=69, n_jobs=-1)
rf_orig.fit(X_train, y_train)
importances = rf_orig.feature_importances_

ax.bar(features, importances, color='forestgreen')
ax.set_ylabel('Importância')
ax.set_title('Importância das Features Originais - Random Forest')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def show_tree_window():
    # Permite navegar entre as 25 árvores do Random Forest treinado (rf)
    tree_root = tk.Tk()
    tree_root.title("Árvores do Random Forest (depth=2)")

    current_tree_idx = tk.IntVar(value=0)

    fig, ax = plt.subplots(figsize=(10, 5))
    canvas = FigureCanvasTkAgg(fig, master=tree_root)
    canvas.get_tk_widget().pack(fill='both', expand=True)

    def draw_tree(idx):
        ax.clear()
        plot_tree(
            rf.estimators_[idx],
            feature_names=features,
            filled=True,
            rounded=True,
            max_depth=2,
            fontsize=10,
            ax=ax
        )
        ax.set_title(f"Árvore {idx+1} do Random Forest (depth=2)")
        plt.tight_layout()
        canvas.draw()

    def prev_tree():
        idx = current_tree_idx.get()
        if idx > 0:
            current_tree_idx.set(idx - 1)
            draw_tree(idx - 1)

    def next_tree():
        idx = current_tree_idx.get()
        if idx < len(rf.estimators_) - 1:
            current_tree_idx.set(idx + 1)
            draw_tree(idx + 1)

    btn_frame = tk.Frame(tree_root)
    btn_frame.pack(fill='x', expand=False)
    btn_prev = tk.Button(btn_frame, text="Anterior", command=prev_tree)
    btn_prev.pack(side='left', padx=10, pady=5)
    btn_next = tk.Button(btn_frame, text="Próxima", command=next_tree)
    btn_next.pack(side='right', padx=10, pady=5)

    draw_tree(0)

    def on_tree_closing():
        tree_root.destroy()

    tree_root.protocol("WM_DELETE_WINDOW", on_tree_closing)
    tree_root.mainloop()

def on_closing_bar():
    root.destroy()
    show_tree_window()

root.protocol("WM_DELETE_WINDOW", on_closing_bar)
root.mainloop()