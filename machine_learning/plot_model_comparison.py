import matplotlib.pyplot as plt
import numpy as np

# Para o nosso caso de classificação (prever se um jogo é caro/barato), a métrica mais importante geralmente é o F1-score.
# Isso porque o F1-score equilibra precisão e recall, sendo mais robusto quando as classes estão desbalanceadas
# ou quando tanto falsos positivos quanto falsos negativos são relevantes para a análise.
# Use F1-score como principal métrica para comparar modelos de classificação neste projeto.
#
# Entre os modelos avaliados, o KNN apresentou o maior F1-score (0.95), sendo portanto o melhor modelo para classificação neste contexto,
# pois consegue equilibrar bem precisão e recall, superando os demais modelos nas métricas principais.

# Dados de classificação
class_models = ['KNN', 'Naive Bayes', 'SVM (linear)', 'SVM (rbf)', 'SVM (poly)', 'Logistic Regression']
accuracy = [0.95, 0.56, 0.57, 0.60, 0.58, 0.56]
precision = [0.96, 0.55, 0.55, 0.57, 0.55, 0.55]
recall = [0.95, 0.64, 0.73, 0.90, 0.97, 0.58]
f1 = [0.95, 0.59, 0.63, 0.70, 0.70, 0.57]

# Gráfico de barras para classificação
x = np.arange(len(class_models))
width = 0.2

plt.figure(figsize=(10,6))
plt.bar(x - 1.5*width, accuracy, width, label='Acurácia')
plt.bar(x - 0.5*width, precision, width, label='Precisão')
plt.bar(x + 0.5*width, recall, width, label='Recall')
plt.bar(x + 1.5*width, f1, width, label='F1-score')
plt.xticks(x, class_models, rotation=20)
plt.ylabel('Score')
plt.title('Comparação de Métricas de Classificação')
plt.legend()
plt.tight_layout()
plt.show()

# Para o nosso caso de regressão (prever o valor dos jogos), a métrica mais importante geralmente é o RMSE.
# O RMSE penaliza mais fortemente grandes erros e é fácil de interpretar pois está na mesma unidade do valor previsto.
# Entre os modelos avaliados, o Random Forest apresentou o menor RMSE (4.03), sendo portanto o melhor modelo para regressão neste contexto,
# pois apresenta o menor erro médio nas previsões de preço dos jogos.

# Dados de regressão
reg_models = ['Linear Regression', 'Ridge', 'Lasso', 'Decision Tree (5)', 'Decision Tree (20)', 'Random Forest', 'Neural Net (Single)', 'Neural Net (Multi)']
rmse = [11.38, 11.34, 11.38, 9.07, 4.78, 4.03, 9.45, 9.31]
mae = [4.47, 4.32, 4.48, 4.14, 2.31, 0.25, 4.16, 4.15]
r2 = [0.01, 0.01, 0.01, 0.11, 0.75, 0.88, 0.03, 0.06]

# Gráfico de barras para RMSE
plt.figure(figsize=(10,6))
plt.bar(reg_models, rmse, color='skyblue')
plt.ylabel('RMSE')
plt.title('Comparação de RMSE dos Modelos de Regressão')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# O gráfico de MAE mostra o erro absoluto médio dos modelos de regressão.
# Valores menores de MAE indicam previsões mais próximas dos valores reais, sem penalizar tanto grandes erros quanto o RMSE.
# O Random Forest também apresenta o menor MAE (0.25), reforçando sua qualidade para prever o valor dos jogos.

# Gráfico de barras para MAE
plt.figure(figsize=(10,6))
plt.bar(reg_models, mae, color='orange')
plt.ylabel('MAE')
plt.title('Comparação de MAE dos Modelos de Regressão')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# O gráfico de R² mostra o quanto cada modelo explica da variabilidade dos preços dos jogos.
# Quanto mais próximo de 1, melhor. O Random Forest atinge o maior R² (0.88), mostrando excelente capacidade de explicação dos dados.

# Gráfico de barras para R²
plt.figure(figsize=(10,6))
plt.bar(reg_models, r2, color='green')
plt.ylabel('R²')
plt.title('Comparação de R² dos Modelos de Regressão')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# Gráfico de score médio cross-validation
# O gráfico de score médio de cross-validation permite comparar o desempenho geral dos modelos em diferentes tarefas.
# No nosso caso, o KNN e o Random Forest apresentam os maiores scores médios, indicando desempenho consistente em validação cruzada.
# Isso reforça que são boas escolhas para classificação e regressão, respectivamente, no nosso conjunto de dados.

# Gráfico de score médio cross-validation
cv_models = ['KMeans', 'DecisionTreeClassifier', 'KNN', 'LinearRegression', 'LogisticRegression', 'RandomForest', 'Ridge', 'Lasso', 'NaiveBayes', 'NeuralNetworks', 'SVM']
cv_scores = [0.4433, 0.6256, 0.9517, 0.0195, 0.5575, 0.9473, 0.0195, 0.0194, 0.5578, 0.0303, 0.6013]

plt.figure(figsize=(10,6))
plt.bar(cv_models, cv_scores, color='purple')
plt.ylabel('Score Médio')
plt.title('Score Médio de Cross-validation por Modelo')
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()



