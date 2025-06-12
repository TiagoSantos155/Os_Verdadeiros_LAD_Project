# Sistema de Recomendação: Content-Based Filtering e SVD (Matrix Factorization)

Este projeto utiliza dois métodos principais para recomendar jogos aos utilizadores: **Content-Based Filtering** e **Matrix Factorization (SVD)**. Abaixo explicamos detalhadamente como cada abordagem funciona no contexto deste sistema.

---

## 1. Content-Based Filtering

### O que é?

O Content-Based Filtering recomenda jogos com base nas preferências explícitas do utilizador, analisando os géneros dos jogos que o utilizador gosta ou avaliou positivamente. O sistema assume que, se um utilizador gosta de certos géneros, provavelmente irá gostar de outros jogos com géneros semelhantes.

### Como funciona neste projeto?

1. **Identificação dos géneros preferidos:**
   - O sistema analisa as avaliações (`ratings`) feitas pelo utilizador a jogos anteriores.
   - Para cada avaliação, extrai os géneros associados ao jogo e soma as pontuações dadas pelo utilizador a cada género.
   - Calcula-se a média das avaliações por género para identificar os géneros favoritos.
   - Se o utilizador nunca avaliou jogos, utiliza os géneros escolhidos no registo.

2. **Seleção de jogos recomendados:**
   - Para cada género preferido, o sistema procura jogos desse género no dataset (`simpleDataSet.csv`).
   - Exclui jogos já presentes na biblioteca do utilizador ou já recomendados.
   - Seleciona até um limite de jogos por género (tipicamente 10).
   - Os jogos recomendados são apresentados agrupados por género.

3. **Vantagens e limitações:**
   - **Vantagem:** Não depende de avaliações de outros utilizadores, apenas das preferências do próprio.
   - **Limitação:** Pode ser limitado se o utilizador não tiver avaliações ou se os géneros preferidos forem pouco representados no dataset.

### Exemplo de fluxo:

- O utilizador avalia positivamente jogos de "Action" e "RPG".
- O sistema identifica "Action" e "RPG" como géneros preferidos.
- Recomenda jogos desses géneros que o utilizador ainda não possui.

---

## 2. Matrix Factorization (SVD)

### O que é?

Matrix Factorization, usando SVD (Singular Value Decomposition), é uma técnica de **collaborative filtering**. Em vez de olhar apenas para o conteúdo dos jogos, analisa padrões de avaliação entre todos os utilizadores para prever que jogos um utilizador pode gostar, mesmo que nunca os tenha avaliado.

### Como funciona neste projeto?

1. **Construção da matriz de avaliações:**
   - Cria-se uma matriz `R` onde cada linha representa um utilizador e cada coluna um jogo.
   - Cada célula contém a avaliação (rating) que o utilizador deu ao jogo, ou zero se nunca avaliou.

2. **Decomposição SVD:**
   - Aplica-se SVD à matriz `R`, decompondo-a em três matrizes: `U`, `S`, `Vt`.
   - Mantém-se apenas os `k` fatores latentes principais (tipicamente até 20), reduzindo a dimensionalidade.
   - Reconstrói-se uma matriz aproximada `R_hat` que estima as avaliações que cada utilizador daria a cada jogo.

3. **Geração de recomendações:**
   - Para o utilizador atual, identifica jogos que ainda não avaliou.
   - Ordena esses jogos pela avaliação prevista em `R_hat`.
   - Recomenda os jogos com maior pontuação prevista.

4. **Vantagens e limitações:**
   - **Vantagem:** Capaz de encontrar padrões complexos de gosto, mesmo entre utilizadores com gostos diferentes.
   - **Limitação:** Requer que existam avaliações suficientes no sistema para funcionar bem (problema do "cold start").

### Exemplo de fluxo:

- O utilizador nunca avaliou "Game X", mas utilizadores com gostos semelhantes gostaram desse jogo.
- O SVD prevê uma pontuação alta para "Game X" para este utilizador.
- O sistema recomenda "Game X".

---

## 3. Integração no Projeto

- **Na página principal do utilizador**, o sistema tenta primeiro recomendar jogos usando Matrix Factorization (SVD).
- Se não houver dados suficientes (ex: utilizador novo), recorre ao Content-Based Filtering, usando os géneros preferidos.
- O sistema também mostra recomendações por género e os jogos mais populares.

---

## 4. Código Relacionado

- **Content-Based Filtering:** Funções `get_preferred_genres`, `get_games_by_genre` em `backend/app.py`.
- **Matrix Factorization (SVD):** Função `get_matrix_factorization_recommendations` em `backend/app.py`.

