from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import os

app = Flask(__name__)

USERS_CSV_PATH = "../dataset/users.csv"
DEVS_CSV_PATH = "../dataset/developers.csv"

@app.route('/')
def index():
    return render_template('login.html')

@app.route('/login', methods=['POST'])
def login():
    username = request.form['username']
    password = request.form['password']

    users_df = pd.read_csv(USERS_CSV_PATH)

    # Verificar se o usuário e senha existem
    match = users_df[(users_df['username'] == username) & (users_df['password'] == password)]

    if not match.empty:
        return f"Login bem-sucedido! Bem-vindo, {username}"
    else:
        return "Login falhou. Verifica os dados."

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        if not os.path.exists(USERS_CSV_PATH):
            # Cria ficheiro se não existir
            users_df = pd.DataFrame(columns=['user_id', 'username', 'password'])
        else:
            users_df = pd.read_csv(USERS_CSV_PATH)

        # Verifica se username já existe
        if username in users_df['username'].values:
            return "Nome de utilizador já existe. Escolhe outro."

        # Criar novo user_id
        new_user_id = users_df['user_id'].max() + 1 if not users_df.empty else 1

        # Adicionar novo utilizador
        new_user = pd.DataFrame([{
            'user_id': new_user_id,
            'username': username,
            'password': password
        }])

        users_df = pd.concat([users_df, new_user], ignore_index=True)
        users_df.to_csv(USERS_CSV_PATH, index=False)

        return redirect(url_for('index'))

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

if __name__ == '__main__':
    app.run(debug=True)
