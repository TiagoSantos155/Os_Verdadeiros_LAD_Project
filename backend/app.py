from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import pandas as pd
import os
import requests
import json
from functools import lru_cache
import numpy as np
import warnings
import sys
import joblib

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

sys.path.append(os.path.join(os.path.dirname(__file__), '../Modelos'))
from predict_price import load_model, predict_with_model

# Caminho para os encoders/scaler do Random Forest
RF_ENCODERS_PATH = os.path.join(BASE_DIR, "../Modelos/random_forest_encoders.pkl")
RF_MODEL_PATH = os.path.join(BASE_DIR, "../Modelos/random_forest_model.pkl")

def rf_predict_price(developer, genres, release_date):
    rf = joblib.load(RF_MODEL_PATH)
    encoders_scaler = joblib.load(RF_ENCODERS_PATH)
    label_encoders = encoders_scaler["label_encoders"]
    scaler = encoders_scaler["scaler"]
    features = encoders_scaler["features"]

    dev_val = developer

    # Garante que cada género está limpo e monta a string no formato do dataset
    genres_clean = [g.strip() for g in genres if g.strip()]
    if len(genres_clean) == 1:
        genres_val = f"['{genres_clean[0]}']"
    else:
        genres_val = "[" + ", ".join(f"'{g}'" for g in genres_clean) + "]"

    release_val = pd.to_datetime(release_date, errors='coerce')
    release_ord = release_val.toordinal() if pd.notnull(release_val) else 0

    dev_encoded = label_encoders['developers'].transform([dev_val])[0] if dev_val in label_encoders['developers'].classes_ else 0

    # Tenta prever para a combinação, senão tenta para cada género individual (no formato do encoder)
    if genres_val in label_encoders['genres'].classes_:
        genres_encoded = label_encoders['genres'].transform([genres_val])[0]
        X = [[dev_encoded, genres_encoded, release_ord]]
        X_scaled = scaler.transform(X)
        pred = rf.predict(X_scaled)[0]
        if pred < 1:
            pred = 0
        return float(pred)
    else:
        preds = []
        for g in genres_clean:
            g_val = f"['{g}']"
            if g_val in label_encoders['genres'].classes_:
                genres_encoded = label_encoders['genres'].transform([g_val])[0]
                X = [[dev_encoded, genres_encoded, release_ord]]
                X_scaled = scaler.transform(X)
                pred = rf.predict(X_scaled)[0]
                if pred < 1:
                    pred = 0
                preds.append(float(pred))
        if preds:
            return float(np.mean(preds))
        raise ValueError(
            f"A combinação de géneros '{genres_val}' não existe no modelo. "
            f"Géneros disponíveis: {list(label_encoders['genres'].classes_)}"
        )

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

    # Biblioteca do utilizador
    user_library = []
    if 'library' in user_row and pd.notna(user_row['library']) and str(user_row['library']).strip():
        try:
            import ast
            user_library = ast.literal_eval(user_row['library'])
            if not isinstance(user_library, list):
                user_library = []
        except Exception:
            user_library = []
    else:
        user_library = []

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
        genre_recommendations=genre_recommendations,
        user_library=user_library
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
            return redirect(url_for('developer_home'))  # Redireciona para a página do developer
        else:
            # Renderiza o template com erro e mantém o username preenchido
            return render_template('developer_login.html', login_error=True, request=request)

    return render_template('developer_login.html', login_error=False, request=request)

@app.route('/developer-home')
def developer_home():
    if 'dev_username' not in session:
        return redirect(url_for('developer_login'))
    dev_username = session['dev_username']
    # Lista de 12 botões e imagens associadas
    buttons = [
        {"id": "btn1", "label": "Preço vs Popularidade", "img": url_for('static', filename='dev_images/preco_pop.png')},
        {"id": "btn2", "label": "Top10 Developers", "img": url_for('static', filename='dev_images/dev_populares.png')},
        {"id": "btn3", "label": "Países com mais jogadores", "img": url_for('static', filename='dev_images/top10_paises.png')},
        {"id": "btn4", "label": "Jogos mais caros", "img": url_for('static', filename='dev_images/jogos_caros.png')},
        {"id": "btn5", "label": "Jogos mais jogados", "img": url_for('static', filename='dev_images/mais_jogados.png')},
        {"id": "btn6", "label": "Jogos mais lucrativos", "img": url_for('static', filename='dev_images/muito_dinheiro.png')},

    ]
    return render_template('developer_home.html', dev_username=dev_username, buttons=buttons)

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

# Lista de géneros igual à usada no treino do modelo (ajusta conforme necessário)
ALL_GENRES = [
    "Action", "Adventure", "RPG", "Strategy", "Simulation", "Sports", "Indie", "Puzzle", "Racing", "Shooter"
    # ...adiciona todos os géneros usados no treino...
]

@app.route('/predict_price_api', methods=['POST'])
def predict_price_api():
    try:
        data = request.get_json()
        title = data.get('title', '')
        release_date = data.get('release_date', '')
        genres = data.get('genres', '')
        developer = "unknown"
        genres_selected = [g.strip() for g in genres.split(',') if g.strip()]
        predicted_price = rf_predict_price(developer, genres_selected, release_date)
        return jsonify(success=True, predicted_price=round(predicted_price, 2))
    except ValueError as ve:
        return jsonify(success=False, error=str(ve))
    except Exception as e:
        import traceback
        print("Erro no /predict_price_api:", e)
        print(traceback.format_exc())
        return jsonify(success=False, error=str(e))

@app.route('/optimize_price_api', methods=['POST'])
def optimize_price_api():
    try:
        data = request.get_json()
        title = data.get('title', '')
        release_date = data.get('release_date', '')
        genres = data.get('genres', '')
        developer = "unknown"
        genres_selected = [g.strip() for g in genres.split(',') if g.strip()]
        predicted_price = rf_predict_price(developer, genres_selected, release_date)
        euros = int(predicted_price)
        cents = predicted_price - euros
        if cents >= 0.80:
            cents = 0.99
        elif cents >= 0.50:
            cents = 0.79
        else:
            cents = 0.49
        optimized_price = round(euros + cents, 2)
        return jsonify(success=True, optimized_price=optimized_price)
    except ValueError as ve:
        return jsonify(success=False, error=str(ve))
    except Exception as e:
        import traceback
        print("Erro no /optimize_price_api:", e)
        print(traceback.format_exc())
        return jsonify(success=False, error=str(e))

def ensure_ratings_column():
    users_df = pd.read_csv(USERS_CSV_PATH)
    if 'ratings' not in users_df.columns:
        users_df['ratings'] = '{}'
        users_df.to_csv(USERS_CSV_PATH, index=False)
ensure_ratings_column()

def ensure_library_column():
    users_df = pd.read_csv(USERS_CSV_PATH)
    if 'library' not in users_df.columns:
        users_df['library'] = '[]'
        users_df.to_csv(USERS_CSV_PATH, index=False)
ensure_library_column()

def get_user_library(username):
    users_df = pd.read_csv(USERS_CSV_PATH)
    user_row = users_df[users_df['username'] == username]
    if user_row.empty:
        return []
    library_str = user_row.iloc[0].get('library', '[]')
    try:
        library = json.loads(library_str)
    except Exception:
        library = []
    return library

def save_user_library(username, library):
    users_df = pd.read_csv(USERS_CSV_PATH)
    idx = users_df[users_df['username'] == username].index
    if len(idx) == 0:
        return
    idx = idx[0]
    users_df.at[idx, 'library'] = json.dumps(library)
    users_df.to_csv(USERS_CSV_PATH, index=False)

@app.route('/add_to_library', methods=['POST'])
def add_to_library():
    if 'username' not in session:
        return {"success": False, "error": "Unauthorized"}, 401
    username = session['username']
    data = request.json
    gameid = int(data.get('gameid'))
    library = get_user_library(username)
    if gameid not in library:
        library.append(gameid)
        save_user_library(username, library)
    return {"success": True}

@app.route('/library')
def library():
    if 'username' not in session:
        return redirect(url_for('login'))
    username = session['username']
    library_gameids = get_user_library(username)
    if not library_gameids:
        games = []
    else:
        df = pd.read_csv(SIMPLE_DATASET_CSV)
        games = []
        for gid in library_gameids:
            row = df[df['gameid'] == gid]
            if not row.empty:
                games.append({
                    "gameid": int(gid),
                    "title": row.iloc[0]['title'],
                    "img": get_game_image_cached(int(gid))
                })
    return render_template('biblioteca.html', username=username, games=games)

@app.route('/developer-add-game', methods=['GET', 'POST'])
def developer_add_game():
    if 'dev_username' not in session:
        return redirect(url_for('developer_login'))

    genres_list = get_all_genres()
    predicted_price = None
    optimized_price = None

    if request.method == 'POST':
        title = request.form.get('title', '').strip()
        release_date = request.form.get('release_date', '').strip()
        genres = request.form.get('genres', '').strip()
        action = request.form.get('action')
        if not title or not release_date or not genres:
            flash("Preencha todos os campos e selecione pelo menos um género.", "error")
        elif action == "optimize":
            # --- Prever preço base usando Random Forest ---
            developer = "unknown"
            genres_selected = [g.strip() for g in genres.split(',') if g.strip()]
            try:
                predicted_price = rf_predict_price(developer, genres_selected, release_date)
            except Exception as e:
                predicted_price = 0
            # --- Otimização do preço ---
            euros = int(predicted_price)
            cents = predicted_price - euros
            if cents >= 0.80:
                cents = 0.99
            elif cents >= 0.50:
                cents = 0.79
            else:
                cents = 0.49
            optimized_price = round(euros + cents, 2)
            flash(f"Preço otimizado sugerido: {optimized_price:.2f} € (preço previsto: {predicted_price:.2f} €)", "success")

    return render_template(
        'developer_add_game.html',
        genres=genres_list,
        predicted_price=optimized_price if optimized_price is not None else None
    )

# Certifica que a coluna 'ratings' existe no users.csv
def ensure_ratings_column():
    users_df = pd.read_csv(USERS_CSV_PATH)
    if 'ratings' not in users_df.columns:
        users_df['ratings'] = '{}'
        users_df.to_csv(USERS_CSV_PATH, index=False)
ensure_ratings_column()

if __name__ == '__main__':
    app.run(debug=True)