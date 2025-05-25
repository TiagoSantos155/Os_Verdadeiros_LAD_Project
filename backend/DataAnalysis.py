import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import ttkbootstrap as ttk  # Modern themes for tkinter
import ast 
import matplotlib.ticker as ticker
import mplcursors
from ttkbootstrap.constants import *
from tkinter import messagebox
from PIL import Image, ImageTk  # For image handling
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 1- Import and read the CSV file
file_path = "purchased_games_final.csv"
df = pd.read_csv(file_path)

def show_summary_statistics():
    """Displays a table with summary statistics for numerical columns in the dataset."""
    if df is not None and not df.empty:
        numerical_df = df.select_dtypes(include=[np.number])

        if not numerical_df.empty:
            summary_stats = numerical_df.describe().transpose().reset_index()
            summary_stats.rename(columns={'index': 'Coluna'}, inplace=True)
            summary_stats = summary_stats.drop(columns=['25%', '50%', '75%'], errors='ignore')
            summary_stats = summary_stats.round(2)

            # Nova janela
            table_window = ttk.Toplevel()
            table_window.title("Resumo Estatístico")
            table_window.geometry("800x400")
            table_window.protocol("WM_DELETE_WINDOW", table_window.destroy)

            # Frame para o Treeview e scrollbars
            frame = ttk.Frame(table_window)
            frame.pack(fill="both", expand=True)

            # Scrollbars
            vsb = ttk.Scrollbar(frame, orient="vertical")
            hsb = ttk.Scrollbar(frame, orient="horizontal")

            # Treeview
            tree = ttk.Treeview(
                frame,
                columns=list(summary_stats.columns),
                show="headings",
                bootstyle="info",
                yscrollcommand=vsb.set,
                xscrollcommand=hsb.set
            )

            # Scrollbars configuradas
            vsb.config(command=tree.yview)
            hsb.config(command=tree.xview)
            vsb.pack(side="right", fill="y")
            hsb.pack(side="bottom", fill="x")

            # Cabeçalhos e colunas
            for col in summary_stats.columns:
                tree.heading(col, text=col)
                tree.column(col, anchor="center", width=120)

            # Inserir os dados
            for _, row in summary_stats.iterrows():
                tree.insert("", "end", values=row.tolist())

            tree.pack(fill="both", expand=True)

        else:
            messagebox.showinfo("Informação", "Não há colunas numéricas no DataFrame.")
    else:
        messagebox.showerror("Erro", "O DataFrame está vazio ou não foi carregado.")

def plot_top20_games():
    """Creates a bar chart for the top 20 most played games."""
    if df is not None:
        if 'title' in df.columns and 'playerid' in df.columns:
            top20 = df.groupby('title')['playerid'].nunique().nlargest(20).reset_index()
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top20, x='playerid', y='title', palette='viridis')
            plt.title('Top 20 Jogos Mais Jogados')
            plt.xlabel('Número de Jogadores Únicos')
            plt.ylabel('Nome dos Jogos')
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Erro", "As colunas 'title' e/ou 'playerid' não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def show_top_achievement_games():
    """Displays a table of games with the highest average percentage of achievements in the interface."""
    if df is not None:
        if {'title', 'playerid', 'achievements_unlocked', 'total_achievements'}.issubset(df.columns):
            filtered_games = df.groupby('title').filter(lambda x: x['playerid'].nunique() >= 1000)
            if not filtered_games.empty:
                filtered_games['achievement_percentage'] = (
                    filtered_games['achievements_unlocked'] / filtered_games['total_achievements'] * 100
                )
                top_games = (
                    filtered_games.groupby('title')['achievement_percentage']
                    .mean()
                    .nlargest(10)
                    .reset_index()
                )

                # Create a new window to display the table
                table_window = ttk.Toplevel()
                table_window.title("Jogos com Maior Percentagem de Conquistas")
                table_window.geometry("600x400")

                # Create a Treeview widget to display the table
                tree = ttk.Treeview(
                    table_window, columns=("title", "achievement_percentage"), show="headings", bootstyle="info"
                )
                tree.heading("title", text="Nome do Jogo")
                tree.heading("achievement_percentage", text="Percentagem Média de Conquistas")
                tree.column("title", anchor="center", width=300)
                tree.column("achievement_percentage", anchor="center", width=200)

                for _, row in top_games.iterrows():
                    tree.insert("", "end", values=(row['title'], f"{row['achievement_percentage']:.2f}%"))

                tree.pack(fill="both", expand=True)
            else:
                messagebox.showinfo("Informação", "Nenhum jogo com pelo menos 1000 jogadores foi encontrado.")
        else:
            messagebox.showerror("Erro", "As colunas necessárias não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def show_top20_players():
    """Displays a table of the top 20 players with the most games."""
    if df is not None:
        if {'playerid', 'title', 'release_date'}.issubset(df.columns):
            # Convert release_date to datetime if not already
            if not np.issubdtype(df['release_date'].dtype, np.datetime64):
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

            # Group by playerid and calculate the total number of games
            player_stats = df.groupby('playerid').agg(
                total_games=('title', 'count'),
                most_recent_game=('release_date', 'max')
            ).reset_index()

            # Merge to get the title of the most recent game
            player_stats = player_stats.merge(
                df[['playerid', 'title', 'release_date']],
                left_on=['playerid', 'most_recent_game'],
                right_on=['playerid', 'release_date'],
                how='left'
            ).rename(columns={'title': 'most_recent_title'})

            # Sort by total games and select the top 20 players
            top20_players = player_stats.nlargest(20, 'total_games')[['playerid', 'most_recent_title', 'most_recent_game', 'total_games']]
            top20_players['most_recent_game'] = top20_players['most_recent_game'].dt.year  # Extract year

            # Create a new window to display the table
            table_window = ttk.Toplevel()
            table_window.title("Top 20 Jogadores com Mais Jogos")
            table_window.geometry("700x400")

            # Create a Treeview widget to display the table
            tree = ttk.Treeview(
                table_window, columns=("playerid", "most_recent_title", "most_recent_game", "total_games"), show="headings", bootstyle="info"
            )
            tree.heading("playerid", text="ID do Jogador")
            tree.heading("most_recent_title", text="Jogo Mais Recente")
            tree.heading("most_recent_game", text="Ano")
            tree.heading("total_games", text="Número de Jogos")
            tree.column("playerid", anchor="center", width=150)
            tree.column("most_recent_title", anchor="center", width=250)
            tree.column("most_recent_game", anchor="center", width=100)
            tree.column("total_games", anchor="center", width=150)

            # Insert data into the Treeview
            for _, row in top20_players.iterrows():
                tree.insert("", "end", values=(row['playerid'], row['most_recent_title'], row['most_recent_game'], row['total_games']))

            tree.pack(fill="both", expand=True)
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('playerid', 'title', 'release_date') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def show_top20_spenders():
    """Displays a table of the top 20 players who spent the most money."""
    if df is not None:
        if {'playerid', 'title', 'release_date', 'eur'}.issubset(df.columns):
            # Convert release_date to datetime if not already
            if not np.issubdtype(df['release_date'].dtype, np.datetime64):
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

            # Group by playerid and calculate total spending and most expensive game
            player_spending = df.groupby('playerid').agg(
                total_spent=('eur', 'sum'),
                most_expensive_game_cost=('eur', 'max')
            ).reset_index()

            # Merge to get the title and release year of the most expensive game
            player_spending = player_spending.merge(
                df[['playerid', 'title', 'release_date', 'eur']],
                left_on=['playerid', 'most_expensive_game_cost'],
                right_on=['playerid', 'eur'],
                how='left'
            ).rename(columns={'title': 'most_expensive_title', 'release_date': 'most_expensive_year'})

            # Extract the year from the release date
            player_spending['most_expensive_year'] = player_spending['most_expensive_year'].dt.year

            # Sort by total spending and select the top 20 players
            top20_spenders = player_spending.nlargest(20, 'total_spent')[['playerid', 'most_expensive_title', 'most_expensive_year', 'most_expensive_game_cost', 'total_spent']]

            # Create a new window to display the table
            table_window = ttk.Toplevel()
            table_window.title("Top 20 Jogadores que Gastaram Mais Dinheiro")
            table_window.geometry("800x400")

            # Create a Treeview widget to display the table
            tree = ttk.Treeview(
                table_window, columns=("playerid", "most_expensive_title", "most_expensive_year", "most_expensive_game_cost", "total_spent"), show="headings", bootstyle="info"
            )
            tree.heading("playerid", text="ID do Jogador")
            tree.heading("most_expensive_title", text="Jogo Mais Caro")
            tree.heading("most_expensive_year", text="Ano")
            tree.heading("most_expensive_game_cost", text="Custo do Jogo (€)")
            tree.heading("total_spent", text="Gasto Total (€)")
            tree.column("playerid", anchor="center", width=150)
            tree.column("most_expensive_title", anchor="center", width=250)
            tree.column("most_expensive_year", anchor="center", width=100)
            tree.column("most_expensive_game_cost", anchor="center", width=150)
            tree.column("total_spent", anchor="center", width=150)

            # Insert data into the Treeview
            for _, row in top20_spenders.iterrows():
                tree.insert("", "end", values=(row['playerid'], row['most_expensive_title'], row['most_expensive_year'], f"{row['most_expensive_game_cost']:.2f}€", f"{row['total_spent']:.2f}€"))

            tree.pack(fill="both", expand=True)
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('playerid', 'title', 'release_date', 'eur') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def show_top20_avg_spenders():
    """Displays a table of the top 20 players with the highest average spending per paid game."""
    if df is not None:
        if {'playerid', 'title', 'release_date', 'eur'}.issubset(df.columns):
            # Filter only paid games (price > 0)
            paid_games = df[df['eur'] > 0]

            # Convert release_date to datetime if not already
            if not np.issubdtype(paid_games['release_date'].dtype, np.datetime64):
                paid_games['release_date'] = pd.to_datetime(paid_games['release_date'], errors='coerce')

            # Group by playerid to calculate the average spending on paid games
            player_spending = paid_games.groupby('playerid').agg(
                avg_spent=('eur', 'mean'),
                most_expensive_game_cost=('eur', 'max')
            ).reset_index()

            # Merge to get the title and release year of the most expensive game
            player_spending = player_spending.merge(
                paid_games[['playerid', 'title', 'release_date', 'eur']],
                left_on=['playerid', 'most_expensive_game_cost'],
                right_on=['playerid', 'eur'],
                how='left'
            ).rename(columns={'title': 'most_expensive_title', 'release_date': 'most_expensive_year'})

            # Extract the year from the release date
            player_spending['most_expensive_year'] = player_spending['most_expensive_year'].dt.year

            # Sort by average spending and select the top 20 players
            top20_avg_spenders = player_spending.nlargest(20, 'avg_spent')[['playerid', 'most_expensive_title', 'most_expensive_year', 'most_expensive_game_cost', 'avg_spent']]

            # Create a new window to display the table
            table_window = ttk.Toplevel()
            table_window.title("Top 20 Jogadores com Maior Gasto Médio por Jogo Pago")
            table_window.geometry("800x400")

            # Create a Treeview widget to display the table
            tree = ttk.Treeview(
                table_window, columns=("playerid", "most_expensive_title", "most_expensive_year", "most_expensive_game_cost", "avg_spent"), show="headings", bootstyle="info"
            )
            tree.heading("playerid", text="ID do Jogador")
            tree.heading("most_expensive_title", text="Jogo Mais Caro")
            tree.heading("most_expensive_year", text="Ano")
            tree.heading("most_expensive_game_cost", text="Custo do Jogo (€)")
            tree.heading("avg_spent", text="Gasto Médio (€)")
            tree.column("playerid", anchor="center", width=150)
            tree.column("most_expensive_title", anchor="center", width=250)
            tree.column("most_expensive_year", anchor="center", width=100)
            tree.column("most_expensive_game_cost", anchor="center", width=150)
            tree.column("avg_spent", anchor="center", width=150)

            # Insert data into the Treeview
            for _, row in top20_avg_spenders.iterrows():
                tree.insert("", "end", values=(row['playerid'], row['most_expensive_title'], row['most_expensive_year'], f"{row['most_expensive_game_cost']:.2f}€", f"{row['avg_spent']:.2f}€"))

            tree.pack(fill="both", expand=True)
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('playerid', 'title', 'release_date', 'eur') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def show_top20_achievement_players():
    """Exibe os 20 jogadores com melhor média e total de conquistas desbloqueadas."""
    if df is not None:
        if {'playerid', 'title', 'achievements_unlocked', 'total_achievements'}.issubset(df.columns):
            # Agrupar por jogador para calcular média e total de conquistas
            player_achievements = df.groupby('playerid').agg(
                avg_achievements=('achievements_unlocked', 'mean'),
                total_achievements_unlocked=('achievements_unlocked', 'sum')
            ).reset_index()

            # Selecionar os top 20 com maior média
            top20 = player_achievements.nlargest(20, 'avg_achievements')

            # Obter o jogo onde cada jogador desbloqueou mais conquistas
            max_game = df.loc[df.groupby('playerid')['achievements_unlocked'].idxmax()]
            top20 = top20.merge(
                max_game[['playerid', 'title', 'achievements_unlocked']],
                on='playerid',
                how='left'
            ).rename(columns={
                'title': 'game_with_most_achievements',
                'achievements_unlocked': 'max_achievements_in_game'
            })

            # Criar nova janela com tabela
            table_window = ttk.Toplevel()
            table_window.title("Top 20 Jogadores com Melhor Média de Conquistas")
            table_window.geometry("1000x450")

            # Criar tabela (Treeview)
            tree = ttk.Treeview(
                table_window,
                columns=("playerid", "game_with_most_achievements", "max_achievements_in_game", "avg_achievements", "total_achievements_unlocked"),
                show="headings",
                bootstyle="info"
            )

            # Cabeçalhos
            tree.heading("playerid", text="ID do Jogador")
            tree.heading("game_with_most_achievements", text="Jogo com Mais Conquistas")
            tree.heading("max_achievements_in_game", text="Conquistas no Jogo")
            tree.heading("avg_achievements", text="Média de Conquistas")
            tree.heading("total_achievements_unlocked", text="Total de Conquistas")

            # Largura e alinhamento
            tree.column("playerid", anchor="center", width=120)
            tree.column("game_with_most_achievements", anchor="center", width=300)
            tree.column("max_achievements_in_game", anchor="center", width=120)
            tree.column("avg_achievements", anchor="center", width=150)
            tree.column("total_achievements_unlocked", anchor="center", width=150)

            # Inserir dados
            for _, row in top20.iterrows():
                tree.insert("", "end", values=(
                    row['playerid'],
                    row['game_with_most_achievements'],
                    row['max_achievements_in_game'],
                    f"{row['avg_achievements']:.2f}",
                    int(row['total_achievements_unlocked'])
                ))

            tree.pack(fill="both", expand=True, padx=10, pady=10)

        else:
            messagebox.showerror("Erro", "As colunas necessárias ('playerid', 'title', 'achievements_unlocked', 'total_achievements') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def show_top20_genre_diversity():
    """Displays a table of the top 20 players with the most diverse genres played."""
    if df is not None:
        if {'playerid', 'genres'}.issubset(df.columns):
            # Split genres into individual entries if they are comma-separated
            df['genres'] = df['genres'].fillna('').str.split(',')

            # Explode the genres column to have one genre per row
            exploded_df = df.explode('genres')

            # Group by playerid and calculate the number of unique genres played
            player_genres = exploded_df.groupby('playerid').agg(
                num_genres=('genres', 'nunique'),
                favorite_genre=('genres', lambda x: x.value_counts().idxmax())
            ).reset_index()

            # Sort by the number of genres and select the top 20 players
            top20_genre_diversity = player_genres.nlargest(20, 'num_genres')

            # Create a new window to display the table
            table_window = ttk.Toplevel()
            table_window.title("Top 20 Jogadores com Maior Diversidade de Géneros")
            table_window.geometry("700x400")

            # Create a Treeview widget to display the table
            tree = ttk.Treeview(
                table_window, columns=("playerid", "num_genres", "favorite_genre"), show="headings", bootstyle="info"
            )
            tree.heading("playerid", text="ID do Jogador")
            tree.heading("num_genres", text="Número de Géneros Jogados")
            tree.heading("favorite_genre", text="Género Favorito")
            tree.column("playerid", anchor="center", width=150)
            tree.column("num_genres", anchor="center", width=200)
            tree.column("favorite_genre", anchor="center", width=250)

            # Insert data into the Treeview
            for _, row in top20_genre_diversity.iterrows():
                tree.insert("", "end", values=(row['playerid'], row['num_genres'], row['favorite_genre']))

            tree.pack(fill="both", expand=True)
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('playerid', 'genres') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def formatar_valores(valor, _):
    """Formata os valores do eixo Y (ex: 1000 -> 1K, 1000000 -> 1M)."""
    if valor >= 1_000_000:
        return f'{valor/1_000_000:.1f}M'
    elif valor >= 1_000:
        return f'{valor/1_000:.1f}K'
    else:
        return str(int(valor))

def plot_games_per_year():
    """Plota o número de jogos únicos lançados por ano, com interatividade ao passar o mouse."""

    if df is not None:
        if 'release_date' in df.columns and 'gameid' in df.columns:

            # Limpeza e deduplicação
            df_valid = df.dropna(subset=['gameid', 'release_date'])
            df_valid['gameid'] = df_valid['gameid'].astype(str)
            df_unique = df_valid.drop_duplicates(subset='gameid')
            df_unique['release_date'] = pd.to_datetime(df_unique['release_date'], errors='coerce')
            df_unique = df_unique.dropna(subset=['release_date'])
            df_unique['release_year'] = df_unique['release_date'].dt.year

            print(f"Total de jogos únicos considerados: {len(df_unique)}")

            # Todos os anos
            anos_validos = df_unique['release_year'].dropna().astype(int)
            todos_os_anos = pd.Series(range(anos_validos.min(), anos_validos.max() + 1), name='release_year')

            # Contagem por ano
            games_per_year = df_unique.groupby('release_year').size().reset_index(name='num_games')
            games_per_year = todos_os_anos.to_frame().merge(games_per_year, on='release_year', how='left').fillna(0)
            games_per_year['num_games'] = games_per_year['num_games'].astype(int)

            # Plot
            plt.figure(figsize=(12, 7))
            lineplot = sns.lineplot(data=games_per_year, x='release_year', y='num_games', marker='o', color='blue')
            plt.title('Número de Jogos Únicos Lançados por Ano')
            plt.xlabel('Ano')
            plt.ylabel('Número de Jogos')
            plt.grid(True)

            # Formatador do eixo Y
            ax = plt.gca()
            ax.yaxis.set_major_formatter(ticker.FuncFormatter(formatar_valores))

            # Linha de referência (ex: 2017)
            plt.axvline(x=2017, color='red', linestyle='--', linewidth=1.5)

            # Interatividade ao passar o rato (hover)
            cursor = mplcursors.cursor(lineplot.lines[0], hover=True)
            @cursor.connect("add")
            def on_add(sel):
                x_val = int(sel.target[0])
                y_val = int(sel.target[1])
                sel.annotation.set_text(f"Ano: {x_val}\nJogos: {y_val}")
                sel.annotation.get_bbox_patch().set(fc="white", alpha=0.9)

            plt.tight_layout()
            plt.show()

        else:
            messagebox.showerror("Erro", "O DataFrame precisa conter as colunas 'release_date' e 'gameid'.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def plot_price_vs_release_year():
    """Creates a line chart showing the average price of games by release year with a 3-year gap on the x-axis."""
    if df is not None:
        if {'release_date', 'eur'}.issubset(df.columns):
            # Convert release_date to datetime if not already
            if not np.issubdtype(df['release_date'].dtype, np.datetime64):
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

            # Extract the year from the release_date
            df['release_year'] = df['release_date'].dt.year

            # Group by release year and calculate the average price
            price_per_year = df.groupby('release_year')['eur'].mean().reset_index().rename(columns={'eur': 'avg_price'})

            # Plot the line chart
            plt.figure(figsize=(10, 6))
            sns.lineplot(data=price_per_year, x='release_year', y='avg_price', marker='o', color='g')
            plt.title('Preço Médio dos Jogos por Ano de Lançamento', fontsize=16)
            plt.xlabel('Ano de Lançamento', fontsize=12)
            plt.ylabel('Preço Médio (€)', fontsize=12)
            plt.grid(True)

            # Ensure min_year and max_year are integers
            min_year = int(price_per_year['release_year'].min())
            max_year = int(price_per_year['release_year'].max())

            # Set x-axis ticks with a 3-year gap
            plt.xticks(range(min_year, max_year + 1, 3), fontsize=10)
            plt.yticks(fontsize=10)

            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('release_date', 'eur') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def plot_players_per_genre():
    """Creates a bar chart showing the number of players for the top 25 genres."""
    if df is not None:
        if {'genres', 'playerid'}.issubset(df.columns):
            # Split genres into individual entries if they are comma-separated
            df['genres'] = df['genres'].fillna('').str.split(',')
            exploded_df = df.explode('genres')

            # Group by genre and count the number of unique players
            players_per_genre = exploded_df.groupby('genres')['playerid'].nunique().reset_index().rename(columns={'playerid': 'num_players'})

            # Sort by the number of players, filter out empty genres, and select the top 25
            players_per_genre = players_per_genre[players_per_genre['genres'] != ''].nlargest(25, 'num_players')

            # Plot the bar chart
            plt.figure(figsize=(10, 6))
            sns.barplot(data=players_per_genre, x='num_players', y='genres', palette='viridis', orient='h')
            plt.title('Top 25 Géneros com Mais Jogadores', fontsize=16)
            plt.xlabel('Número de Jogadores', fontsize=12)
            plt.ylabel('Género', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('genres', 'playerid') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def plot_price_vs_popularity_by_genre():
    """Cria um gráfico de dispersão mostrando a relação entre preço médio e popularidade para os 20 principais gêneros."""
    if df is not None:
        if {'genres', 'eur', 'playerid'}.issubset(df.columns):
            df_clean = df.copy()

            # Converte strings como "['action']" em listas reais
            df_clean['genres'] = df_clean['genres'].fillna('').apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x
            )

            # Garante que tudo é lista (evita valores únicos como strings soltas)
            df_clean['genres'] = df_clean['genres'].apply(
                lambda x: [x] if isinstance(x, str) else x
            )

            # Limpa os gêneros: tira espaços e força minúsculas
            df_clean['genres'] = df_clean['genres'].apply(
                lambda lst: [g.strip().lower() for g in lst if isinstance(g, str) and g.strip()]
            )

            exploded_df = df_clean.explode('genres')

            # Agrupamento por gênero
            genre_stats = exploded_df.groupby('genres').agg(
                avg_price=('eur', 'mean'),
                total_players=('playerid', 'nunique')
            ).reset_index()

            # Seleciona os 20 gêneros mais populares
            genre_stats = genre_stats.nlargest(20, 'total_players')

            # Plotagem
            plt.figure(figsize=(12, 7))
            sns.set(style="whitegrid")

            palette = sns.color_palette("hsv", len(genre_stats))

            sns.scatterplot(
                data=genre_stats,
                x='avg_price',
                y='total_players',
                hue='genres',
                palette=palette,
                s=120,
                edgecolor='black'
            )

            plt.title('Preço vs Popularidade (Top 20 Géneros)', fontsize=16)
            plt.xlabel('Preço Médio (€)', fontsize=12)
            plt.ylabel('Número Total de Jogadores', fontsize=12)
            plt.legend(title='Género', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('genres', 'eur', 'playerid') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def show_top10_developers():
    """Displays a table of the top 10 developers with the most popular games."""
    if df is not None:
        # Check if the required columns exist in the DataFrame
        missing_columns = {'developers', 'title', 'playerid'} - set(df.columns)
        if missing_columns:
            messagebox.showerror("Erro", f"As colunas necessárias estão em falta: {', '.join(missing_columns)}")
            return

        # Group by developers and calculate the total number of unique players across all their games
        developer_stats = df.groupby('developers').agg(
            total_players=('playerid', 'nunique'),
            total_games=('title', 'nunique')
        ).reset_index()

        # Sort by total players and select the top 10 developers
        top10_developers = developer_stats.nlargest(10, 'total_players')

        # Create a new window to display the table
        table_window = ttk.Toplevel()
        table_window.title("Top 10 Developers com Jogos Mais Populares")
        table_window.geometry("600x400")

        # Create a Treeview widget to display the table
        tree = ttk.Treeview(
            table_window, columns=("developers", "total_players", "total_games"), show="headings", bootstyle="info"
        )
        tree.heading("developers", text="Developers")
        tree.heading("total_players", text="Total de Jogadores")
        tree.heading("total_games", text="Total de Jogos")
        tree.column("developers", anchor="center", width=250)
        tree.column("total_players", anchor="center", width=150)
        tree.column("total_games", anchor="center", width=150)

        # Insert data into the Treeview
        for _, row in top10_developers.iterrows():
            tree.insert("", "end", values=(row['developers'], row['total_players'], row['total_games']))

        tree.pack(fill="both", expand=True)
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def plot_players_per_country():
    """Creates a bar chart showing the number of players for the top 10 countries with the most players."""
    if df is not None:
        if {'country', 'playerid'}.issubset(df.columns):
            # Group by country and count the number of unique players
            players_per_country = df.groupby('country')['playerid'].nunique().reset_index().rename(columns={'playerid': 'num_players'})

            # Sort by the number of players and select the top 10 countries
            top10_countries = players_per_country.nlargest(10, 'num_players')

            # Plot the bar chart
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top10_countries, x='num_players', y='country', palette='viridis', orient='h')
            plt.title('Top 10 Países com Mais Jogadores', fontsize=16)
            plt.xlabel('Número de Jogadores', fontsize=12)
            plt.ylabel('País', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('country', 'playerid') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def plot_top20_profitable_games():
    """Creates a bar chart showing the top 20 most profitable games."""
    if df is not None:
        if {'title', 'playerid', 'eur'}.issubset(df.columns):
            # Group by title to calculate unique players and average price
            game_stats = df.groupby('title').agg(
                unique_players=('playerid', 'nunique'),
                avg_price=('eur', 'mean')
            ).reset_index()

            # Calculate profitability
            game_stats['profitability'] = game_stats['unique_players'] * game_stats['avg_price']

            # Sort by profitability and select the top 20 games
            top20_profitable_games = game_stats.nlargest(20, 'profitability')

            # Plot the bar chart
            plt.figure(figsize=(10, 6))
            sns.barplot(data=top20_profitable_games, x='profitability', y='title', palette='viridis', orient='h')
            plt.title('Top 20 Jogos Mais Lucrativos', fontsize=16)
            plt.xlabel('Lucratividade (€)', fontsize=12)
            plt.ylabel('Nome do Jogo', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('title', 'playerid', 'eur') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def show_top20_expensive_games():
    """Exibe os 20 jogos mais caros em uma janela com tabela."""
    if df is not None:
        if {'title', 'eur'}.issubset(df.columns):
            # Agrupar por título e pegar o maior preço registrado
            max_price_per_game = df.groupby('title').agg(
                max_price=('eur', 'max')
            ).reset_index()

            # Ordenar e pegar os 20 mais caros
            top20 = max_price_per_game.sort_values(by='max_price', ascending=False).head(20)

            # Criar janela nova
            top_window = ttk.Toplevel()  # <-- aqui está o ajuste
            top_window.title("Top 20 Jogos Mais Caros")
            top_window.geometry("550x420")
            top_window.configure(bg="#2c3e50")

            # Criar Treeview
            tree = ttk.Treeview(top_window, columns=("title", "max_price"), show="headings", height=20)
            tree.heading("title", text="Nome do Jogo")
            tree.heading("max_price", text="Maior Preço (€)")

            tree.column("title", anchor="w", width=380)
            tree.column("max_price", anchor="center", width=120)

            for _, row in top20.iterrows():
                tree.insert("", "end", values=(row['title'], f"€{row['max_price']:.2f}"))

            # Barra de rolagem
            scrollbar = ttk.Scrollbar(top_window, orient="vertical", command=tree.yview)
            tree.configure(yscrollcommand=scrollbar.set)

            tree.pack(side="left", fill="both", expand=True, padx=10, pady=10)
            scrollbar.pack(side="right", fill="y")
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('title', 'eur') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")

def plot_best_genre_combinations():
    """Plota combinações de gêneros com melhor relação entre preço médio e número de jogadores."""
    if df is not None and {'genres', 'eur', 'playerid'}.issubset(df.columns):
        df_clean = df.copy()

        # Converte string para lista
        df_clean['genres'] = df_clean['genres'].fillna('').apply(
            lambda x: ast.literal_eval(x) if isinstance(x, str) and x.startswith('[') and x.endswith(']') else x
        )

        # Força tudo a lista
        df_clean['genres'] = df_clean['genres'].apply(
            lambda x: [x] if isinstance(x, str) else x
        )

        # Limpa os gêneros
        df_clean['genres'] = df_clean['genres'].apply(
            lambda lst: sorted([g.strip().lower() for g in lst if isinstance(g, str) and g.strip()])
        )

        # Cria string única para o grupo de géneros
        df_clean['genre_combo'] = df_clean['genres'].apply(lambda x: ', '.join(x))

        combo_stats = df_clean.groupby('genre_combo').agg(
            avg_price=('eur', 'mean'),
            total_players=('playerid', 'nunique'),
            num_games=('genre_combo', 'count')
        ).reset_index()

        # Filtra para combos com pelo menos 10 jogos
        combo_stats = combo_stats[combo_stats['num_games'] >= 10]

        # Pega os 20 combos com mais jogadores
        top_combos = combo_stats.nlargest(20, 'total_players')

        plt.figure(figsize=(14, 8))
        sns.set(style="whitegrid")

        sns.scatterplot(
            data=top_combos,
            x='avg_price',
            y='total_players',
            hue='genre_combo',
            palette='tab20',
            s=150,
            edgecolor='black'
        )

        plt.title('Preço Médio vs Popularidade (Top Combos de Géneros)', fontsize=16)
        plt.xlabel('Preço Médio (€)', fontsize=12)
        plt.ylabel('Número Total de Jogadores', fontsize=12)
        plt.legend(title='Combinação de Géneros', bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.show()
    else:
        messagebox.showerror("Erro", "As colunas necessárias ('genres', 'eur', 'playerid') não existem no DataFrame.")
    
def plot_correlation_matrix():
    """Plota uma matriz de correlação entre os preços e os anos de lançamento."""
    if df is not None:
        if {'release_date', 'eur'}.issubset(df.columns):
            # Converter release_date para datetime se necessário
            if not np.issubdtype(df['release_date'].dtype, np.datetime64):
                df['release_date'] = pd.to_datetime(df['release_date'], errors='coerce')

            # Extrair o ano de lançamento
            df['release_year'] = df['release_date'].dt.year

            # Selecionar apenas as colunas relevantes
            correlation_data = df[['release_year', 'eur']].dropna()

            # Calcular a matriz de correlação
            correlation_matrix = correlation_data.corr()

            # Plotar a matriz de correlação
            plt.figure(figsize=(8, 6))
            sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", cbar=True)
            plt.title('Matriz de Correlação: Preço vs Ano de Lançamento', fontsize=16)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.tight_layout()
            plt.show()
        else:
            messagebox.showerror("Erro", "As colunas necessárias ('release_date', 'eur') não existem no DataFrame.")
    else:
        messagebox.showerror("Erro", "O dataframe está vazio ou não foi carregado.")
    
def create_interface():
    """Creates an enhanced interface using ttkbootstrap."""
    app = ttk.Window(themename="superhero")  # Use a modern theme
    app.title("Análise de Dados de Jogos")
    app.geometry("900x700")

    # Add a header
    header_frame = ttk.Frame(app, padding=10)
    header_frame.pack(fill="x", pady=10)

    title_label = ttk.Label(
        header_frame, text="Análise de Dados de Jogos", font=("Helvetica", 24, "bold"), bootstyle="primary"
    )
    title_label.pack(side="left", padx=10)

    # Add a navigation frame
    nav_frame = ttk.Frame(app, padding=20)
    nav_frame.pack(fill="x", pady=20)

    # Group buttons by type
    table_buttons = [
        ("Resumo Estatístico", show_summary_statistics, "info-outline"),
        ("Jogos com Maior Percentagem de Conquistas", show_top_achievement_games, "info-outline"),
        ("Top 20 Jogadores com Mais Jogos", show_top20_players, "warning-outline"),
        ("Top 20 Jogadores que Gastaram Mais", show_top20_spenders, "danger-outline"),
        ("Top 20 Jogadores com Melhor Média de Conquistas", show_top20_achievement_players, "primary-outline"),
        ("Top 20 Jogadores com Maior Diversidade de Géneros", show_top20_genre_diversity, "info-outline"),
        ("Top 20 Jogadores com Maior Gasto Médio por Jogo Pago", show_top20_avg_spenders, "danger-outline"),
        ("Top 10 Developers com Jogos Mais Populares", show_top10_developers, "info-outline"),
        ("Top 20 Jogos Mais Caros", show_top20_expensive_games, "danger-outline"),
    ]

    graph_buttons = [
        ("Top 20 Jogos Mais Jogados", plot_top20_games, "success-outline"),
        ("Gráfico de Jogos Criados por Ano", plot_games_per_year, "success-outline"),
        ("Gráfico de Preço Médio vs Ano de Lançamento", plot_price_vs_release_year, "info-outline"),
        ("Gráfico de Preço vs Popularidade por Género", plot_price_vs_popularity_by_genre, "warning-outline"),
        ("Gráfico de Jogadores por País", plot_players_per_country, "success-outline"),
        ("Gráfico de Jogos Mais Lucrativos", plot_top20_profitable_games, "warning-outline"),
        ("Gráfico de Combos de Géneros Mais Populares", plot_best_genre_combinations, "primary-outline"),
        ("Matriz de Correlação: Preço vs Ano de Lançamento", plot_correlation_matrix, "info-outline"),
    ]

    # Add table-related buttons
    table_label = ttk.Label(nav_frame, text="Tabelas", font=("Helvetica", 16, "bold"), bootstyle="primary")
    table_label.grid(row=0, column=0, columnspan=5, pady=10)
    for i, (text, command, style) in enumerate(table_buttons):
        row, col = divmod(i, 5)  # Calculate row and column
        ttk.Button(
            nav_frame, text=text, command=command, bootstyle=style
        ).grid(row=row + 1, column=col, padx=10, pady=10)

    # Add graph-related buttons
    graph_label = ttk.Label(nav_frame, text="Gráficos", font=("Helvetica", 16, "bold"), bootstyle="primary")
    graph_label.grid(row=len(table_buttons) // 5 + 2, column=0, columnspan=5, pady=10)
    for i, (text, command, style) in enumerate(graph_buttons):
        row, col = divmod(i, 5)  # Calculate row and column
        ttk.Button(
            nav_frame, text=text, command=command, bootstyle=style
        ).grid(row=row + len(table_buttons) // 5 + 3, column=col, padx=10, pady=10)

    # Add a footer
    footer_label = ttk.Label(
        app,
        text="Desenvolvido por Os Verdadeiros, 2025",
        font=("Helvetica", 10, "italic"),
        bootstyle="inverse-secondary", 
    )
    footer_label.pack(side="bottom", pady=10)

    app.mainloop()

# Call the interface creation function
if __name__ == "__main__":
    create_interface()