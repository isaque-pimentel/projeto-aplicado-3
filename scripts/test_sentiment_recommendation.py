"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides an interactive testing environment for the hybrid recommendation system.
It allows users to generate hybrid recommendations by combining collaborative filtering,
content-based filtering, and sentiment analysis based on user input.
"""

import logging
import os
import sqlite3

import pandas as pd
from googletrans import Translator
from helpers import load_model, print_table, load_similarity_matrix
from textblob import TextBlob

from hybrid_recommendation_system import calculate_hybrid_scores
from sentiment_recommendation import classify_emotion, EMOTION_GENRE_MAP
import traceback

LOG_FILE = "test_sentiment_recommendation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def translate_to_english(text):
    """
    Traduz o texto para inglês usando googletrans. Retorna texto traduzido ou original em caso de erro.
    """
    try:
        translator = Translator()
        translation = translator.translate(text, dest="en")
        return translation.text
    except Exception as e:
        logging.warning(f"Falha ao traduzir texto: {e}")
        return text


def recommend_hybrid_with_emotion(
    algo, ratings_df, movies_df, similarity_df, emotion, user_id=None, n=10, emotion_weight=0.2
):
    """
    Recomenda filmes usando o modelo híbrido, ajustando o score para priorizar gêneros ligados à emoção detectada.
    Se user_id for None ou inválido, faz cold-start.
    """
    genres = EMOTION_GENRE_MAP.get(emotion, EMOTION_GENRE_MAP["neutral"])
    if user_id is not None and user_id in ratings_df["UserID"].values:
        user_ratings = ratings_df[ratings_df["UserID"] == user_id]
        top_n = calculate_hybrid_scores(algo, user_ratings, similarity_df)
        scored = []
        for _, movie_id, score in top_n:
            genres_movie = movies_df[movies_df["MovieID"] == movie_id]["Genres"].values[0]
            boost = emotion_weight if any(g in genres_movie for g in genres) else 0
            scored.append((movie_id, score + boost))
        scored.sort(key=lambda x: x[1], reverse=True)
        recs = pd.DataFrame([
            {
                "MovieID": movie_id,
                "Title": movies_df[movies_df["MovieID"] == movie_id]["Title"].values[0],
                "Genres": movies_df[movies_df["MovieID"] == movie_id]["Genres"].values[0],
                "Score": score,
            }
            for movie_id, score in scored[:n]
        ])
        return recs
    else:
        # Cold-start: recomenda populares do gênero
        filtered = movies_df[movies_df["Genres"].apply(lambda g: any(gen in g for gen in genres))]
        if filtered.empty:
            filtered = movies_df
        return filtered.head(n)


def explain_emotion_to_user(emotion: str):
    """
    Print and log an explanation to the user about how their emotion influenced the recommendation.
    """
    genre_list = EMOTION_GENRE_MAP.get(emotion, EMOTION_GENRE_MAP["neutral"])
    explanation = (
        f"Sua emoção detectada foi: {emotion.capitalize()}. "
        f"Por isso, priorizamos recomendações nos gêneros: {', '.join(genre_list)}. "
        "Se quiser recomendações diferentes, tente descrever seu humor de outra forma."
    )
    print(explanation)
    logging.info(f"User explanation: {explanation}")


if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system testing script.")
    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_paths = [
            os.path.join(project_dir, "models", "svd_movielens_1m_with_recency.pkl"),
            os.path.join(project_dir, "models", "svd_movielens_1m_without_recency.pkl"),
        ]
        sim_path = os.path.join(project_dir, "models", "content_similarity_tfidf.pkl")
        # Load the trained model (usando apenas o primeiro para exemplo)
        try:
            algo = load_model(model_paths[0])
        except Exception as e:
            print("Erro ao carregar o modelo de recomendação.")
            logging.error(f"Erro ao carregar modelo: {e}\n{traceback.format_exc()}")
            exit(1)
        # Load the ratings, movies, and reviews data
        try:
            conn = sqlite3.connect(db_path)
            ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
            movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
            conn.close()
        except Exception as e:
            print("Erro ao carregar dados do banco de dados.")
            logging.error(f"Erro ao carregar dados: {e}\n{traceback.format_exc()}")
            exit(1)
        # Carregar matriz de similaridade (TF-IDF por padrão)
        try:
            similarity_df = load_similarity_matrix(sim_path)
        except Exception as e:
            print("Erro ao carregar matriz de similaridade.")
            logging.error(f"Erro ao carregar matriz de similaridade: {e}\n{traceback.format_exc()}")
            exit(1)
        # Interação com usuário
        user_input = input("Descreva seu humor e o que você gostaria de assistir hoje: ")
        user_input_en = translate_to_english(user_input)
        print(f"\nSua mensagem traduzida para inglês: {user_input_en}")
        logging.info(f"User input (PT): {user_input}")
        logging.info(f"User input (EN): {user_input_en}")
        try:
            emotion = classify_emotion(user_input_en)
            logging.info(f"Detected emotion: {emotion}")
            print(f"\nEmoção detectada: {emotion.capitalize()}")
            explain_emotion_to_user(emotion)
        except Exception as e:
            print("Não foi possível detectar emoção. Mostrando filmes populares.")
            logging.error(f"Emotion detection failed: {e}\n{traceback.format_exc()}")
            emotion = "neutral"
        # Pergunta se o usuário quer associar a recomendação a um UserID
        user_id = None
        try:
            associate = input("Deseja associar a recomendação ao seu histórico? (s/n): ").strip().lower()
            if associate == 's':
                user_id_input = input("Digite seu UserID (ou pressione Enter para ignorar): ").strip()
                if user_id_input:
                    user_id = int(user_id_input)
                    if user_id not in ratings_df["UserID"].values:
                        print("UserID não encontrado. Recomendação será feita sem personalização.")
                        logging.warning(f"UserID informado não encontrado: {user_id}")
                        user_id = None
                else:
                    print("UserID não informado. Recomendação será feita sem personalização.")
        except Exception as e:
            print("Erro ao processar UserID. Recomendação será feita sem personalização.")
            logging.error(f"Erro ao processar UserID: {e}\n{traceback.format_exc()}")
            user_id = None
        # Recomenda usando modelo híbrido com boost de emoção
        try:
            recommendations = recommend_hybrid_with_emotion(
                algo, ratings_df, movies_df, similarity_df, emotion, user_id=user_id, n=10, emotion_weight=0.2
            )
            if recommendations.empty:
                print("Nenhuma recomendação encontrada para seu humor. Mostrando filmes populares.")
                recommendations = movies_df.head(10)
        except Exception as e:
            print("Erro ao gerar recomendações. Mostrando filmes populares.")
            logging.error(f"Erro ao recomendar: {e}\n{traceback.format_exc()}")
            recommendations = movies_df.head(10)
        # Exibe recomendações
        print_table(recommendations, "Recomendações personalizadas baseadas no seu humor")
        # Feedback do usuário
        try:
            feedback = input("Você gostou dessas recomendações? (s/n): ").strip().lower()
            if feedback == 's':
                print("Obrigado pelo seu feedback positivo!")
                logging.info("Usuário aprovou as recomendações.")
            elif feedback == 'n':
                print("Obrigado pelo seu feedback. Vamos trabalhar para melhorar!")
                logging.info("Usuário NÃO aprovou as recomendações.")
            else:
                print("Feedback não reconhecido. Obrigado mesmo assim!")
                logging.info(f"Feedback não reconhecido: {feedback}")
        except Exception as e:
            print("Erro ao registrar feedback.")
            logging.error(f"Erro ao registrar feedback: {e}\n{traceback.format_exc()}")
    except Exception as e:
        print("Erro inesperado. Consulte os logs para mais detalhes.")
        logging.critical(f"Interactive testing failed: {e}\n{traceback.format_exc()}")
