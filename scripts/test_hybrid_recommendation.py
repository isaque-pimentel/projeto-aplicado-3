"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides an interactive testing environment for the hybrid recommendation system.
It allows users to generate and evaluate hybrid recommendations for a given user, using the
same logic and utilities as hybrid_recommendation_system.py.
"""

import logging
import os
import sqlite3
import pandas as pd
from scripts.helpers import (
    calculate_content_similarity,
    load_model,
    print_table,
    load_similarity_matrix,
    save_similarity_matrix,
    get_dynamic_alpha,
    evaluate_recommendations,
)
from scripts.hybrid_recommendation_system import calculate_hybrid_scores

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "test_hybrid_recommendation.log")


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

def show_cold_start_recommendations(ratings_df, movies_df, n=10):
    """Show top-n popular movies for cold-start users."""
    movie_counts = ratings_df["MovieID"].value_counts().head(n)
    recommendations = movies_df[movies_df["MovieID"].isin(movie_counts.index)].copy()
    recommendations["NumRatings"] = recommendations["MovieID"].map(movie_counts)
    recommendations = recommendations.sort_values(by="NumRatings", ascending=False)
    print_table(recommendations, f"Top {n} Popular Movies (Cold Start)")
    print("\n--- End of evaluation for this user ---\n")

def evaluate_and_print_hybrid(algo, ratings_df, movies_df, similarity_df, user_id, n, similarity_method):
    """Evaluate and print hybrid recommendations for a user."""
    user_ratings = ratings_df[ratings_df["UserID"] == user_id]
    if user_ratings.empty:
        show_cold_start_recommendations(ratings_df, movies_df, n=n)
        return
    top_n = calculate_hybrid_scores(algo, user_ratings, similarity_df, alpha_func=get_dynamic_alpha)
    top_n.sort(key=lambda x: x[2], reverse=True)
    top_n = top_n[:n]
    recs = pd.DataFrame([
        {
            "MovieID": movie_id,
            "Title": movies_df[movies_df["MovieID"] == movie_id]["Title"].values[0],
            "Genres": movies_df[movies_df["MovieID"] == movie_id]["Genres"].values[0],
            "HybridScore": score,
        }
        for _, movie_id, score in top_n
    ])
    print_table(recs, f"Top {n} Hybrid Recommendations for User {user_id} ({similarity_method})")
    metrics = evaluate_recommendations(recs, user_ratings, n=n)
    print("\nEvaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.2f}")

if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system testing script.")
    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path = os.path.join(project_dir, "models", "svd_movielens_1m_with_recency.pkl")
        similarity_methods = ["tfidf", "count"]
        sim_paths = [os.path.join(project_dir, "models", f"content_similarity_{m}.pkl") for m in similarity_methods]

        # Load model
        algo = load_model(model_path)

        # Load data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()

        # Load or compute all similarity matrices
        similarity_dfs = []
        for sim_path, method in zip(sim_paths, similarity_methods):
            if os.path.exists(sim_path):
                similarity_df = load_similarity_matrix(sim_path)
            else:
                similarity_df = calculate_content_similarity(movies_df, method=method)
                save_similarity_matrix(similarity_df, sim_path)
            similarity_dfs.append((similarity_df, method))

        while True:
            try:
                user_id_input = input("Enter the User ID for recommendations (or -1 to exit): ")
                user_id = int(user_id_input)
            except ValueError:
                print("Invalid input. Please enter a valid integer User ID or -1 to exit.")
                continue
            if user_id == -1:
                print("Exiting interactive evaluation.")
                break
            try:
                n = int(input("Enter the number of recommendations to display (default 10): ") or 10)
            except ValueError:
                n = 10
            for similarity_df, sim_method in similarity_dfs:
                evaluate_and_print_hybrid(algo, ratings_df, movies_df, similarity_df, user_id, n, sim_method)
            print("\n--- End of evaluation for this user ---\n")
        logging.info("Hybrid recommendation system testing completed successfully.")
    except Exception as e:
        logging.critical("Interactive testing failed: %s", e, exc_info=True)
