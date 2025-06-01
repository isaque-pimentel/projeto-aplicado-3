"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides an interactive testing environment for the hybrid recommendation system.
It allows users to generate hybrid recommendations by combining collaborative filtering,
content-based filtering, and sentiment analysis based on user input.
"""

from sentiment_recommendation import (
    detect_emotions_multi_label,
    explain_emotion_recommendation,
    recommend_movies_multi_emotion,
    get_genre_weights_for_emotions,
    ask_user_to_adjust_emotion_genre,
)
from helpers import load_model, print_table, load_similarity_matrix


def get_user_input_pt() -> str:
    return input("Descreva seu humor e o que você gostaria de assistir hoje: ")


def get_user_id(ratings_df) -> int | None:
    try:
        associate = input("Deseja associar a recomendação ao seu histórico? (s/n): ").strip().lower()
        if associate == 's':
            user_id_input = input("Digite seu UserID (ou pressione Enter para ignorar): ").strip()
            if user_id_input:
                user_id = int(user_id_input)
                if user_id not in ratings_df["UserID"].values:
                    print("UserID não encontrado. Recomendação será feita sem personalização.")
                    return None
                return user_id
            else:
                print("UserID não informado. Recomendação será feita sem personalização.")
    except Exception:
        print("Erro ao processar UserID. Recomendação será feita sem personalização.")
    return None


def collect_user_feedback():
    try:
        feedback = input("Você gostou dessas recomendações? (s/n): ").strip().lower()
        if feedback == 's':
            print("Obrigado pelo seu feedback positivo!")
        elif feedback == 'n':
            print("Obrigado pelo seu feedback. Vamos trabalhar para melhorar!")
        else:
            print("Feedback não reconhecido. Obrigado mesmo assim!")
    except Exception:
        print("Erro ao registrar feedback.")


if __name__ == "__main__":
    import logging, os, sqlite3, traceback
    import pandas as pd
    logging.basicConfig(level=logging.INFO)
    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path = os.path.join(project_dir, "models", "svd_movielens_1m_with_recency.pkl")
        sim_path = os.path.join(project_dir, "models", "content_similarity_tfidf.pkl")
        
        # Load model and data
        algo = load_model(model_path)
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()
        similarity_df = load_similarity_matrix(sim_path)
        
        # User interaction
        user_input = get_user_input_pt()
        emotion_weights, clarification, translation_error = detect_emotions_multi_label(user_input)
        if translation_error:
            print(f"[Aviso de tradução] {translation_error}")
        if clarification:
            print(clarification)
        explanation = explain_emotion_recommendation(emotion_weights)
        print(explanation)
        
        # New: let user adjust emotions/genres
        sorted_emotions = sorted(emotion_weights.items(), key=lambda x: x[1], reverse=True)
        top_emotions = [e for e, w in sorted_emotions if w > 0][:2]
        genre_weights = get_genre_weights_for_emotions(top_emotions)
        emotion_weights, genre_weights = ask_user_to_adjust_emotion_genre(emotion_weights, genre_weights)
        user_id = get_user_id(ratings_df)

        # Pass genre_weights to recommendation
        recommendations = recommend_movies_multi_emotion(movies_df, emotion_weights, n=10)
        print_table(recommendations, "Recomendações personalizadas baseadas no seu humor")
        collect_user_feedback()
    except Exception as e:
        print("Erro inesperado. Consulte os logs para mais detalhes.")
        logging.critical(f"Interactive testing failed: {e}\n{traceback.format_exc()}")
