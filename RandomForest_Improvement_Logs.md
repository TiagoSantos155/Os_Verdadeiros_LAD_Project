# Random Forest - Logs de Melhoramento e Experiências

## Modelo 1.0 (500k linhas)
| Métrica                | Valor      |
|------------------------|------------|
| **RMSE (teste)**       | 4.03       |
| **MAE (teste)**        | 0.25       |
| **R2 (teste)**         | 0.88       |
| **Tempo de treino**    | 9.9996 s   |

---

## Modelo 1.1 - Implementado PCA (5k linhas)
- RMSE no teste: 7.19
- MAE no teste: 3.80
- R2 no teste: 0.28
- Tempo de treino (fit): 0.1444 segundos

---

## Modelo 1.2 - Sem PCA e parâmetros ajustados (5k linhas)
> O modelo ficou pior
- RMSE no teste: 7.58
- MAE no teste: 4.41
- R2 no teste: 0.20
- Tempo de treino (fit): 0.3421 segundos

---

## Modelo 1.3 - Ajustar valores perto de 0 para 0 (Gratuitos) (5k Linhas)
> O modelo não mostrou grande diferença
- RMSE no teste: 7.19
- MAE no teste: 3.80
- R2 no teste: 0.28
- Tempo de treino (fit): 0.1444 segundos

---

## Modelo 1.4 - Utilizar mais 1 coluna (total_achievements)
> O modelo demonstrou pior qualidade
- RMSE no teste: 7.50
- MAE no teste: 3.96
- R2 no teste: 0.21
- Tempo de treino (fit): 0.1565 segundos

---

## Modelo 1.5 - Remover 1 coluna (Genres)
> O modelo ficou bem melhor
- RMSE no teste: 6.94
- MAE no teste: 3.54
- R2 no teste: 0.33
- Tempo de treino (fit): 0.1626 segundos

---

## Modelo 1.6 - Normal mas com 50k Linhas
> Com o aumento de linhas o modelo parece se adaptar melhor com a coluna genres
- RMSE no teste: 4.67
- MAE no teste: 1.69
- R2 no teste: 0.89
- Tempo de treino (fit): 1.6843 segundos

---

## Modelo 1.7 - Sem a coluna genres (50k linhas)
> Com a remoção da coluna genres o modelo ficou pior mostrando uma dependência
- RMSE no teste: 5.27
- MAE no teste: 1.76
- R2 no teste: 0.85
- Tempo de treino (fit): 1.1457 segundos

---

## Modelo 1.8 - Utilizar mais 1 coluna (total_achievements) 50k Linhas
> O modelo demonstrou que com o aumento de linhas o total_achievements provoca pior resultado, confirmando que essas 3 colunas são as únicas importantes
- RMSE no teste: 5.33
- MAE no teste: 1.77
- R2 no teste: 0.85
- Tempo de treino (fit): 1.0441 segundos

---

## Modelo 1.9 - Implementar se o valor for menor que 0,5 passa a ser 0
> O modelo demonstrou que fica um pouco pior
- RMSE no teste: 4.67
- MAE no teste: 1.67
- R2 no teste: 0.88
- Tempo de treino (fit): 1.0464 segundos

---

## Modelo 1.9 - Remover outliers do target
> O modelo cozinhou muito: reduziu o RMSE muito, sacrificando um pouco de R2
- RMSE no teste: 1.59
- MAE no teste: 0.71
- R2 no teste: 0.70
- Tempo de treino (fit): 0.8754 segundos

---

## Estudo para encontrar o melhor número de árvores por floresta

### 5k Linhas:
Top 3 melhores valores (menor RMSE):
- n_estimators=80: RMSE=3.07, MAE=2.01, R2=0.38
- n_estimators=82: RMSE=3.07, MAE=2.01, R2=0.38
- n_estimators=83: RMSE=3.07, MAE=2.01, R2=0.38

### 50k Linhas:
Top 3 melhores valores (menor RMSE):
- n_estimators=100: RMSE=1.52, MAE=0.69, R2=0.72
- n_estimators=98: RMSE=1.52, MAE=0.69, R2=0.72
- n_estimators=99: RMSE=1.53, MAE=0.69, R2=0.72

> Aumentar o número de árvores melhora o resultado, mas o tempo de treino aumenta consideravelmente. Solução custo/efetiva: 25 árvores.

#### 5k Linhas
- RMSE no teste: 3.42
- MAE no teste: 2.28
- R2 no teste: 0.27
- Tempo de treino (fit): 0.0637 segundos

#### 50k Linhas
- RMSE no teste: 1.58
- MAE no teste: 0.72
- R2 no teste: 0.69
- Tempo de treino (fit): 0.2987 segundos

> O resultado demonstra um upgrade em RMSE e diminui o tempo de treino para 1/4

---

## Estudo Para Encontrar o Número Otimizado de Depth

### 5k Linhas (1 a 50 depth)
Top 3 melhores valores (menor RMSE):
- max_depth=22: RMSE=3.12, MAE=2.04, R2=0.36, Tempo treino=0.0542s
- max_depth=23: RMSE=3.12, MAE=2.04, R2=0.36, Tempo treino=0.0550s
- max_depth=16: RMSE=3.12, MAE=2.07, R2=0.36, Tempo treino=0.0825s

### 50k Linhas (1 a 50 depth)
Top 3 melhores valores (menor RMSE):
- max_depth=31: RMSE=1.54, MAE=0.71, R2=0.71, Tempo treino=0.2679s
- max_depth=42: RMSE=1.54, MAE=0.71, R2=0.71, Tempo treino=0.2455s
- max_depth=43: RMSE=1.54, MAE=0.71, R2=0.71, Tempo treino=0.2326s

> O RMSE não muda consideravelmente, mas o tempo de treino aumenta quase linearmente. O depth com bom balanço é 43.

- max_depth=43: RMSE=1.54, MAE=0.71, R2=0.71, Tempo treino=0.2326s

- RMSE no teste: 1.58
- MAE no teste: 0.72
- R2 no teste: 0.69
- Tempo de treino (fit): 0.2912 segundos

---

## Observações Finais

- O PCA aumentava o RMSE, mas diminuía o tempo de treino. Decidiu-se retirar o PCA no modelo final.
- Resultado final sem PCA:
    - RMSE no teste: 1.49
    - MAE no teste: 0.68
    - R2 no teste: 0.72
    - Tempo de treino (fit): 0.2446 segundos

---

## Modelo 2.0 (Atual)

**Descrição detalhada do modelo atual:**

- **Pré-processamento:**
  - Utiliza as colunas `developers`, `genres` e `release_date` como features.
  - Codificação de variáveis categóricas (`developers` e `genres`) usando LabelEncoder.
  - Conversão da coluna `release_date` para valores ordinais.
  - Remoção de outliers do target (`eur`) usando o método IQR.
  - Normalização dos dados com StandardScaler.
  - Não utiliza PCA (mantém todas as features originais).

- **Divisão dos dados:**
  - Split em treino, validação e teste (70% treino, 15% validação, 15% teste).

- **Modelo:**
  - RandomForestRegressor com:
    - `n_estimators=25` (25 árvores)
    - `max_depth=43` (profundidade máxima otimizada)
    - `random_state=69` para reprodutibilidade
    - `n_jobs=-1` para uso de todos os núcleos

- **Pós-processamento:**
  - Valores previstos abaixo de 1 são ajustados para 0 (tratando jogos gratuitos).

- **Visualização:**
  - Gráfico de importância das features usando Tkinter e Matplotlib.
  - Interface para navegar entre as árvores individuais da floresta.

- **Resultados típicos (50k linhas, sem PCA, com outlier removal, depth otimizado):**
    - RMSE no teste: ~1.49
    - MAE no teste: ~0.68
    - R2 no teste: ~0.72
    - Tempo de treino (fit): ~0.24 segundos

**Resumo:**  
O modelo 2.0 representa a versão mais robusta e eficiente até agora, equilibrando custo computacional, interpretabilidade e desempenho. Ele utiliza apenas as features mais relevantes, remove outliers, normaliza os dados e ajusta os hiperparâmetros principais (número de árvores e profundidade máxima) com base em experimentação sistemática. O PCA foi removido após análise, pois degradava o desempenho do modelo neste contexto.

---
