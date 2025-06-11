# RealGames - Plataforma de Recomendação de Jogos

## Descrição

**RealGames** é uma plataforma web para recomendação personalizada de jogos, inspirada na Steam, que utiliza técnicas de Machine Learning para sugerir jogos aos utilizadores com base nas suas preferências, avaliações e histórico. O projeto inclui também um painel para developers com análises de dados do mercado de jogos.

## Funcionalidades Principais

- **Registo e Login de Utilizadores e Developers**
- **Seleção de Géneros Preferidos**
- **Recomendações Personalizadas de Jogos**
  - Matrix Factorization (SVD-like) para recomendações colaborativas
  - Recomendações baseadas em conteúdo (géneros preferidos)
- **Avaliação de Jogos (1 a 5 estrelas)**
- **Adicionar Jogos à Biblioteca Pessoal**
- **Visualização dos Jogos Mais Jogados e Top 10**
- **Painel de Developer com Análises de Dados**
  - Gráficos e tabelas sobre preços, popularidade, géneros, países, developers, etc.
- **Interface Gráfica Moderna (Tkinter, Bootstrap, Orbitron Font)**
- **Machine Learning:**
  - Random Forest, Linear Regression, Logistic Regression, Decision Trees, SVM, etc.
  - Scripts para análise, treino e comparação de modelos

## Estrutura do Projeto

```
Os_Verdadeiros_LAD_Project/
│
├── backend/
│   ├── app.py                 # Backend Flask principal
│   ├── DataAnalysis.py        # Scripts de análise de dados (Tkinter/ttkbootstrap)
│   ├── templates/             # Templates HTML (Jinja2)
│   └── static/                # Ficheiros estáticos (CSS, JS, imagens)
│
├── machine_learning/
│   ├── RandomForest.py
│   ├── LinearRegression.py
│   ├── LogisticRegression.py
│   ├── DecisionTrees.py
│   └── cross_validation_all_models.py
│
├── dataset/
│   ├── purchased_games_final.csv   # Dataset principal (LFS)
│   ├── simpleDataSet.csv          # Dataset simplificado
│   ├── users.csv                  # Utilizadores registados
│   └── developers.csv             # Developers registados
│
├── RandomForest_Improvement_Logs.md  # Logs de experimentação ML
├── .gitignore
├── .gitattributes
└── README.md
```

## Como Executar

### 1. Pré-requisitos

- Python 3.8+
- Instalar dependências:
  ```
  pip install flask pandas numpy scikit-learn matplotlib seaborn ttkbootstrap mplcursors
  ```

### 2. Preparar os Dados

- Coloque os ficheiros `purchased_games_final.csv`, `simpleDataSet.csv`, `users.csv` e `developers.csv` na pasta `dataset/`.
- O ficheiro `purchased_games_final.csv` é grande e está versionado via Git LFS.

### 3. Executar o Backend

```bash
cd backend
python app.py
```

- Aceda a [http://localhost:5000](http://localhost:5000) no browser.

### 4. Scripts de Machine Learning

- Os scripts em `machine_learning/` podem ser executados individualmente para treinar e avaliar modelos.
- Exemplo:
  ```bash
  python machine_learning/RandomForest.py
  ```

- Os resultados e logs de experimentação estão em `RandomForest_Improvement_Logs.md`.

### 5. Painel de Análise de Dados

- Execute `backend/DataAnalysis.py` para abrir uma interface gráfica com múltiplas análises e gráficos interativos.

## Principais Tecnologias

- **Backend:** Flask, pandas, numpy
- **Frontend:** HTML, CSS (Bootstrap, Orbitron), JS
- **Machine Learning:** scikit-learn
- **Visualização:** matplotlib, seaborn, Tkinter, ttkbootstrap

## Créditos

Desenvolvido por **Os Verdadeiros** (2025).

---
