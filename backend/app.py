from flask import Flask, render_template, request, redirect, url_for, session
import pandas as pd
import os
import requests

app = Flask(__name__)
app.secret_key = 'um_segredo_simples'  # Necessário para usar sessão

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
USERS_CSV_PATH = os.path.join(BASE_DIR, "../dataset/users.csv")
DEVS_CSV_PATH = os.path.join(BASE_DIR, "../dataset/developers.csv")
GAMES_CSV = os.path.join(BASE_DIR, "../dataset/simpleDataSet.csv")

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

def get_top10_games():
    import pandas as pd
    top10 = []
    try:
        df = pd.read_csv(GAMES_CSV)
        # Agora usa a coluna 'title' em vez de 'gamename'
        for name in TOP10_NAMES:
            row = df[df['title'].str.lower() == name.lower()]
            if not row.empty:
                gameid = int(row.iloc[0]['gameid'])
                # Buscar imagem via API Steam
                api_url = f"https://store.steampowered.com/api/appdetails?appids={gameid}"
                try:
                    resp = requests.get(api_url, timeout=3)
                    data = resp.json()
                    img_url = ""
                    if data and str(gameid) in data and data[str(gameid)]['success']:
                        img_url = data[str(gameid)]['data'].get('header_image', '')
                    top10.append({
                        "name": name,
                        "gameid": gameid,
                        "img": img_url
                    })
                except Exception:
                    top10.append({
                        "name": name,
                        "gameid": gameid,
                        "img": ""
                    })
            else:
                top10.append({
                    "name": name,
                    "gameid": None,
                    "img": ""
                })
    except Exception:
        # Se não conseguir ler o CSV, devolve lista vazia
        return []
    return top10

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

    genres_list = [
        "Ação", "Aventura", "RPG", "Estratégia", "Simulação",
        "Desporto", "Corridas", "Puzzle", "Terror", "Multijogador"
    ]

    if request.method == 'POST':
        selected_genres = request.form.getlist('genres')
        if len(selected_genres) != 3:
            return render_template('select_genres.html', genres=genres_list, error="Seleciona exatamente 3 géneros.")

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
    recommended_games = []  # Placeholder para recomendações

    # Obter Top 10 jogos
    top10_games = get_top10_games()

    return render_template('home.html', username=username, recommended_games=recommended_games, top10_games=top10_games)

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

if __name__ == '__main__':
    app.run(debug=True)
