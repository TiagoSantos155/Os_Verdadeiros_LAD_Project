import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
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
df = df.head(50000)

cols = ['developers', 'genres', 'eur', 'release_date']
data = df[cols].copy()

# Definir o que é "caro" (acima da mediana)
price_threshold = data['eur'].median()
data['is_expensive'] = (data['eur'] > price_threshold).astype(int)

features = ['developers', 'genres', 'release_date']
X = data[features]
y = data['is_expensive']

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

# Decision Tree Classifier com profundidade 5 (critério Gini)
fit_start = time.time()
dtree_5 = DecisionTreeClassifier(max_depth=5, criterion='gini', random_state=69)
dtree_5.fit(X_train_scaled, y_train)
fit_end = time.time()
train_time_5 = fit_end - fit_start

y_pred_5 = dtree_5.predict(X_test_scaled)
accuracy_5 = accuracy_score(y_test, y_pred_5)
precision_5 = precision_score(y_test, y_pred_5, zero_division=0)
recall_5 = recall_score(y_test, y_pred_5, zero_division=0)
f1_5 = f1_score(y_test, y_pred_5, zero_division=0)

results.append(('Depth 5', accuracy_5, precision_5, recall_5, f1_5, train_time_5))

# Desenhar a árvore de profundidade 5 (legível e mais compacta)
plt.figure(figsize=(12, 6))
plot_tree(
    dtree_5,
    feature_names=features,
    filled=True,
    rounded=True,
    max_depth=2,           # Mostra só até profundidade 2 para maior legibilidade
    fontsize=11,
    proportion=True,       # Mostra proporção de amostras em cada nó
    impurity=True,         # Mostra o gini
    class_names=["Barato", "Caro"]
)
plt.title("Árvore de Decisão Classificação (max_depth=5, mostrando até profundidade 2)")
plt.tight_layout()
plt.show()

# Desenhar a árvore completa (cuidado: pode ser muito grande e ilegível!)
plt.figure(figsize=(24, 12))
plot_tree(
    dtree_5,
    feature_names=features,
    filled=True,
    rounded=True,
    max_depth=None,        # Mostra toda a profundidade da árvore
    fontsize=8,
    proportion=True,
    impurity=True,
    class_names=["Barato", "Caro"]
)
plt.title("Árvore de Decisão Classificação (profundidade completa)")
plt.tight_layout()
plt.show()

# Exibir texto da árvore (primeiros níveis)
tree_rules = export_text(dtree_5, feature_names=features, max_depth=3)
print("Regras da árvore (até profundidade 3):")
print(tree_rules)

# Mostrar resultados no terminal
for name, acc, prec, rec, f1, t in results:
    print(f"--- {name} ---")
    print(f"Acurácia no teste: {acc:.2f}")
    print(f"Precisão: {prec:.2f}")
    print(f"Recall: {rec:.2f}")
    print(f"F1-score: {f1:.2f}")
    print(f"Tempo de treino (fit): {t:.4f} segundos")
    print()

# Gráfico comparativo usando Tkinter
root = tk.Tk()
root.title("Decision Tree Classifier Results")

fig, ax = plt.subplots()
metrics = ['Acurácia', 'Precisão', 'Recall', 'F1-score']
values = [accuracy_5, precision_5, recall_5, f1_5]
ax.bar(metrics, values, color=['blue', 'orange', 'green', 'red'])
ax.set_ylim(0, 1)
ax.set_ylabel('Valor')
ax.set_title('Métricas Decision Tree (Depth 5)')

canvas = FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack()

def on_closing():
    root.destroy()
    exit(0)

root.protocol("WM_DELETE_WINDOW", on_closing)
root.mainloop()

# Desenhar várias árvores lado a lado (por exemplo, profundidade 2, 3 e 5)
from sklearn.tree import DecisionTreeClassifier

depths = [2, 3, 5]
trees = []
for d in depths:
    tree = DecisionTreeClassifier(max_depth=d, criterion='gini', random_state=69)
    tree.fit(X_train_scaled, y_train)
    trees.append(tree)

plt.figure(figsize=(18, 5))
for i, (tree, d) in enumerate(zip(trees, depths)):
    plt.subplot(1, len(depths), i + 1)
    plot_tree(
        tree,
        feature_names=features,
        filled=True,
        rounded=True,
        max_depth=d,
        fontsize=8,
        proportion=True,
        impurity=True,
        class_names=["Barato", "Caro"]
    )
    plt.title(f"Profundidade {d}")
plt.tight_layout()
plt.show()

# O que é o Gini?
# O índice de Gini é uma métrica usada em árvores de decisão para classificação.
# Ele mede a "impureza" de um nó: quanto menor o valor, mais puro (mais exemplos de uma só classe).
# Fórmula do Gini para um nó:
# Gini = 1 - sum(p_i^2) para todas as classes i, onde p_i é a proporção de exemplos da classe i no nó.
# - Gini = 0: nó puro (só uma classe)
# - Gini próximo de 0.5: mistura equilibrada de duas classes
# O algoritmo escolhe divisões que minimizam o Gini nos nós filhos.
