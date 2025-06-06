from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import requests
import json
from functools import lru_cache
import numpy as np
import warnings

app = Flask(__name__)
app.secret_key = 'um_segredo_simples'  # Necessário para usar sessão
warnings.filterwarnings("ignore", category=UserWarning)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_CSV_PATH = os.path.join(BASE_DIR, "../dataset/users.csv")
DEVS_CSV_PATH = os.path.join(BASE_DIR, "../dataset/developers.csv")
SIMPLE_DATASET_CSV = os.path.join(BASE_DIR, "../dataset/simpleDataSet.csv")

TOP10_NAMES = [
    "Counter-Strike 2",
    "PUBG: BATTLEGROUNDS",
    "Left 4 Dead 2",
    "PAYDAY 2",
    "Unturned",
    "Apex Legends™",
    "Warframe",
    "Grand Theft Auto V",
    "VR HOT"
]

@lru_cache(maxsize=2048)
def get_game_image_cached(gameid):
    # Tenta obter imagem da API Steam, se falhar tenta buscar do dataset
    api_url = f"https://store.steampowered.com/api/appdetails?appids={gameid}"
    try:
        resp = requests.get(api_url, timeout=3)
        data = resp.json()
        if data and str(gameid) in data and data[str(gameid)]['success']:
            img = data[str(gameid)]['data'].get('header_image', '')
            if img:
                return img
    except Exception:
        pass
    # Fallback: tenta buscar imagem do dataset se existir coluna 'img' ou 'header_image'
    try:
        df = pd.read_csv(SIMPLE_DATASET_CSV)
        row = df[df['gameid'] == gameid]
        if not row.empty:
            if 'img' in row.columns:
                img = row.iloc[0]['img']
                if isinstance(img, str) and img.strip():
                    return img
            if 'header_image' in row.columns:
                img = row.iloc[0]['header_image']
                if isinstance(img, str) and img.strip():
                    return img
    except Exception:
        pass
    return ""

def get_top10_games():
    top10 = []
    try:
        df = pd.read_csv(SIMPLE_DATASET_CSV)
        for name in TOP10_NAMES:
            row = df[df['title'].str.strip().str.lower() == name.strip().lower()]
            if not row.empty:
                gameid = int(row.iloc[0]['gameid'])
                img_url = get_game_image_cached(gameid)
                top10.append({
                    "name": name,
                    "gameid": gameid,
                    "img": img_url
                })
            else:
                top10.append({
                    "name": name,
                    "gameid": None,
                    "img": ""
                })
    except Exception:
        return []
    return top10

def get_games_by_genre(genre, exclude_gameids=None, limit=10):
    df = pd.read_csv(SIMPLE_DATASET_CSV)
    games = []
    exclude_gameids = set(exclude_gameids or [])
    genre_lower = genre.strip().lower()
    for _, row in df.iterrows():
        if int(row['gameid']) in exclude_gameids:
            continue
        genres_str = row['genres']
        genres_list = []
        if isinstance(genres_str, str):
            genres_list = [g.strip().replace("'", "").replace("[", "").replace("]", "") for g in genres_str.replace('"', '').replace("'", "").replace("[", "").replace("]", "").split(",") if g.strip()]
        if any(genre_lower == g.lower() for g in genres_list):
            games.append({
                "gameid": int(row['gameid']),
                "title": row['title'],
                "img": get_game_image_cached(int(row['gameid']))
            })
        if len(games) >= limit:
            break
    return games

def get_user_ratings(username):
    users_df = pd.read_csv(USERS_CSV_PATH)
    user_row = users_df[users_df['username'] == username]
    if user_row.empty:
        return {}
    ratings_str = user_row.iloc[0].get('ratings', '{}')
    try:
        ratings = json.loads(ratings_str)
    except Exception:
        ratings = {}
    return ratings

def save_user_rating(username, gameid, rating, genres):
    users_df = pd.read_csv(USERS_CSV_PATH)
    idx = users_df[users_df['username'] == username].index
    if len(idx) == 0:
        return
    idx = idx[0]
    ratings_str = users_df.at[idx, 'ratings'] if 'ratings' in users_df.columns else '{}'
    try:
        ratings = json.loads(ratings_str)
    except Exception:
        ratings = {}
    ratings[str(gameid)] = {"rating": rating, "genres": genres}
    users_df.at[idx, 'ratings'] = json.dumps(ratings)
    users_df.to_csv(USERS_CSV_PATH, index=False)

def get_preferred_genres(username):
    # Calcula os géneros preferidos com base nas avaliações
    ratings = get_user_ratings(username)
    genre_scores = {}
    genre_counts = {}
    for v in ratings.values():
        for g in v['genres']:
            genre_scores[g] = genre_scores.get(g, 0) + v['rating']
            genre_counts[g] = genre_counts.get(g, 0) + 1
    if not genre_scores:
        # fallback: usar géneros do registo
        users_df = pd.read_csv(USERS_CSV_PATH)
        user_row = users_df[users_df['username'] == username].iloc[0]
        genres = user_row['genres'].split(",")
        return [g.strip() for g in genres][:3]
    # Ordena por média de avaliação
    genre_avg = sorted(genre_scores.items(), key=lambda x: genre_scores[x[0]]/genre_counts[x[0]], reverse=True)
    return [g for g, _ in genre_avg][:3]

def get_all_genres():
    df = pd.read_csv(SIMPLE_DATASET_CSV)
    genres_set = set()
    for genres_str in df['genres']:
        if isinstance(genres_str, str):
            genres_list = [g.strip().replace("'", "").replace("[", "").replace("]", "") for g in genres_str.replace('"', '').replace("'", "").replace("[", "").replace("]", "").split(",") if g.strip()]
            genres_set.update(genres_list)
    return sorted(genres_set)

def get_matrix_factorization_recommendations(username, limit=10):
    """
    Recomenda jogos usando Matrix Factorization (SVD-like) implementado com numpy/pandas.
    Funciona em Python 3.13 (sem surprise).
    """
    users_df = pd.read_csv(USERS_CSV_PATH)
    ratings_data = []
    user_ids = []
    game_ids = []
    for _, row in users_df.iterrows():
        user = row['username']
        ratings_str = row.get('ratings', '{}')
        try:
            ratings = json.loads(ratings_str)
        except Exception:
            ratings = {}
        for gameid, v in ratings.items():
            try:
                rating = int(v['rating'])
                ratings_data.append((user, int(gameid), rating))
                user_ids.append(user)
                game_ids.append(int(gameid))
            except Exception:
                continue

    if not ratings_data:
        return []

    # Criar matriz utilizador-jogo
    user_list = sorted(list(set(user_ids)))
    game_list = sorted(list(set(game_ids)))
    user_idx = {u: i for i, u in enumerate(user_list)}
    game_idx = {g: i for i, g in enumerate(game_list)}
    R = np.zeros((len(user_list), len(game_list)))
    for user, gameid, rating in ratings_data:
        R[user_idx[user], game_idx[gameid]] = rating

    # SVD simples
    try:
        U, s, Vt = np.linalg.svd(R, full_matrices=False)
        k = min(20, len(s))  # número de fatores latentes
        S = np.diag(s[:k])
        U_k = U[:, :k]
        Vt_k = Vt[:k, :]
        R_hat = np.dot(np.dot(U_k, S), Vt_k)
    except Exception:
        return []

    # Recomendar jogos para o utilizador atual
    if username not in user_idx:
        return []
    uidx = user_idx[username]
    user_rated = set()
    for user, gameid, rating in ratings_data:
        if user == username:
            user_rated.add(gameid)

    df_games = pd.read_csv(SIMPLE_DATASET_CSV)
    candidates = []
    for gameid, gidx in game_idx.items():
        if gameid in user_rated:
            continue
        # Previsão de rating
        pred = R_hat[uidx, gidx]
        row = df_games[df_games['gameid'] == gameid]
        if not row.empty:
            candidates.append((pred, {
                "gameid": int(gameid),
                "title": row.iloc[0]['title'],
                "img": get_game_image_cached(int(gameid))
            }))
    candidates.sort(reverse=True, key=lambda x: x[0])
    return [c[1] for c in candidates[:limit]]

@app.route('/')
def info():
    return render_template('info.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        users_df = pd.read_csv(USERS_CSV_PATH)
        # Garante que a coluna genres existe
        if 'genres' not in users_df.columns:
            users_df['genres'] = ''
            users_df.to_csv(USERS_CSV_PATH, index=False)

        match = users_df[(users_df['username'] == username) & (users_df['password'] == password)]

        if not match.empty:
            session['username'] = username  # Guarda o username na sessão
            user_row = match.iloc[0]
            if pd.isna(user_row.get('genres', None)) or user_row.get('genres', '') == '':
                return redirect(url_for('select_genres'))
            return redirect(url_for('home'))
        else:
            return render_template('login.html', login_error=True)
    # Se GET, apenas mostra o formulário
    return render_template('login.html')

@app.route('/select-genres', methods=['GET', 'POST'])
def select_genres():
    if 'username' not in session:
        return redirect(url_for('index'))

    genres_list = get_all_genres()

    if request.method == 'POST':
        selected_genres = request.form.getlist('genres')
        if len(selected_genres) != 3:
            return render_template('select_genres.html', genres=genres_list, error="Select exactly 3 genres.")
        users_df = pd.read_csv(USERS_CSV_PATH)
        username = session['username']
        users_df.loc[users_df['username'] == username, 'genres'] = ','.join(selected_genres)
        users_df.to_csv(USERS_CSV_PATH, index=False)
        return redirect(url_for('home'))

    return render_template('select_genres.html', genres=genres_list)

@app.route('/home')
def home():
    if 'username' not in session:
        return redirect(url_for('index'))
    username = session['username']
    users_df = pd.read_csv(USERS_CSV_PATH)
    user_row = users_df[users_df['username'] == username].iloc[0]
    if pd.isna(user_row.get('genres', None)) or user_row.get('genres', '') == '':
        return redirect(url_for('select_genres'))

    # Recomendações Matrix Factorization (compatível Python 3.13)
    recommended_games = get_matrix_factorization_recommendations(username, limit=10)
    if not recommended_games:
        # fallback para content-based se não houver dados suficientes
        preferred_genres = get_preferred_genres(username)
        shown_gameids = set()
        recommended_games = []
        for genre in preferred_genres:
            games = get_games_by_genre(genre, exclude_gameids=shown_gameids, limit=10)
            for g in games:
                if g['gameid'] not in shown_gameids:
                    recommended_games.append(g)
                    shown_gameids.add(g['gameid'])
            if len(recommended_games) >= 10:
                break
        recommended_games = recommended_games[:10]

    # Top 10 jogos (já implementado)
    top10_games = get_top10_games()

    # Content-based: recomendações por género preferido (secundário)
    preferred_genres = get_preferred_genres(username)
    shown_gameids = set(g['gameid'] for g in top10_games if g['gameid'])
    genre_recommendations = []
    for genre in preferred_genres:
        games = get_games_by_genre(genre, exclude_gameids=shown_gameids, limit=10)
        shown_gameids.update(g['gameid'] for g in games)
        genre_recommendations.append({"genre": genre, "games": games})

    return render_template(
        'home.html',
        username=username,
        recommended_games=recommended_games,
        top10_games=top10_games,
        genre_recommendations=genre_recommendations
    )

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not os.path.exists(USERS_CSV_PATH):
            users_df = pd.DataFrame(columns=['user_id', 'username', 'password', 'genres'])
        else:
            users_df = pd.read_csv(USERS_CSV_PATH)
            if 'genres' not in users_df.columns:
                users_df['genres'] = ''

        if username in users_df['username'].values:
            return "Nome de utilizador já existe. Escolhe outro."

        new_user_id = users_df['user_id'].max() + 1 if not users_df.empty else 1

        new_user = pd.DataFrame([{
            'user_id': new_user_id,
            'username': username,
            'password': password,
            'genres': ''
        }])

        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_CSV_PATH, index=False)

        session['username'] = username
        return redirect(url_for('select_genres'))

    return render_template('register.html')

@app.route('/developer-login', methods=['GET', 'POST'])
def developer_login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not os.path.exists(DEVS_CSV_PATH):
            return "Nenhum developer registado ainda."

        devs_df = pd.read_csv(DEVS_CSV_PATH)
        match = devs_df[(devs_df['username'] == username) & (devs_df['password'] == password)]

        if not match.empty:
            session['dev_username'] = username
            return f"Login de developer bem-sucedido! Bem-vindo, {username}"
        else:
            return "Credenciais de developer inválidas."

    return render_template('developer_login.html')

@app.route('/developer-register', methods=['GET', 'POST'])
def developer_register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not os.path.exists(DEVS_CSV_PATH):
            devs_df = pd.DataFrame(columns=['developer_id', 'username', 'password'])
        else:
            devs_df = pd.read_csv(DEVS_CSV_PATH)

        if username in devs_df['username'].values:
            return "Este developer já existe."

        new_id = devs_df['developer_id'].max() + 1 if not devs_df.empty else 1
        new_dev = pd.DataFrame([{
            'developer_id': new_id,
            'username': username,
            'password': password
        }])

        devs_df = pd.concat([devs_df, new_dev], ignore_index=True)
        devs_df.to_csv(DEVS_CSV_PATH, index=False)

        return redirect(url_for('developer_login'))

    return render_template('developer_register.html')

@app.route('/logout', methods=['GET', 'POST'])
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/rate_game', methods=['POST'])
def rate_game():
    if 'username' not in session:
        return "Unauthorized", 401
    username = session['username']
    data = request.json
    gameid = data.get('gameid')
    rating = int(data.get('rating'))
    # Buscar géneros do jogo
    df = pd.read_csv(SIMPLE_DATASET_CSV)
    row = df[df['gameid'] == int(gameid)]
    genres = []
    if not row.empty:
        genres_str = row.iloc[0]['genres']
        if isinstance(genres_str, str):
            genres = [g.strip().replace("'", "").replace("[", "").replace("]", "") for g in genres_str.replace('"', '').replace("'", "").replace("[", "").replace("]", "").split(",") if g.strip()]
    save_user_rating(username, gameid, rating, genres)
    return {"success": True}

# Certifica que a coluna 'ratings' existe no users.csv
def ensure_ratings_column():
    users_df = pd.read_csv(USERS_CSV_PATH)
    if 'ratings' not in users_df.columns:
        users_df['ratings'] = '{}'
        users_df.to_csv(USERS_CSV_PATH, index=False)
ensure_ratings_column()

if __name__ == '__main__':
    app.run(debug=True)
