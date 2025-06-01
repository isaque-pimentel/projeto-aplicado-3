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
from helpers import (
    calculate_content_similarity,
    load_model,
    print_table,
    load_similarity_matrix,
    save_similarity_matrix,
    get_dynamic_alpha,
    evaluate_recommendations,
)
from hybrid_recommendation_system import calculate_hybrid_scores

LOG_FILE = "test_hybrid_recommendation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)

if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system testing script.")
    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path = os.path.join(project_dir, "models", "svd_movielens_1m_with_recency.pkl")
        similarity_method = "tfidf"  # or "count"
        sim_path = os.path.join(project_dir, "models", f"content_similarity_{similarity_method}.pkl")

        # Load model
        algo = load_model(model_path)

        # Load data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()

        # Load or compute similarity matrix
        if os.path.exists(sim_path):
            similarity_df = load_similarity_matrix(sim_path)
        else:
            similarity_df = calculate_content_similarity(movies_df, method=similarity_method)
            save_similarity_matrix(similarity_df, sim_path)

        # Interactive user input
        user_id = int(input("Enter the User ID for recommendations: "))
        user_ratings = ratings_df[ratings_df["UserID"] == user_id]
        # If user is new (cold start)
        if user_ratings.empty:
            print("User not found or no ratings. Showing most popular movies.")
            # Recommend most popular movies
            movie_counts = ratings_df["MovieID"].value_counts().head(10)
            recommendations = movies_df[movies_df["MovieID"].isin(movie_counts.index)].copy()
            recommendations["NumRatings"] = recommendations["MovieID"].map(movie_counts)
            recommendations = recommendations.sort_values(by="NumRatings", ascending=False)
            print_table(recommendations, f"Top 10 Popular Movies (Cold Start)")
        else:
            # Generate hybrid recommendations for this user
            # Use dynamic alpha for cold start handling
            top_n = calculate_hybrid_scores(algo, user_ratings, similarity_df, alpha_func=get_dynamic_alpha)
            # Sort and get top 10
            top_n.sort(key=lambda x: x[2], reverse=True)
            top_10 = top_n[:10]
            recs = pd.DataFrame([
                {
                    "MovieID": movie_id,
                    "Title": movies_df[movies_df["MovieID"] == movie_id]["Title"].values[0],
                    "Genres": movies_df[movies_df["MovieID"] == movie_id]["Genres"].values[0],
                    "HybridScore": score,
                }
                for _, movie_id, score in top_10
            ])
            print_table(recs, f"Top 10 Hybrid Recommendations for User {user_id}")
            # Evaluate recommendations
            metrics = evaluate_recommendations(recs, user_ratings, n=10)
            print("\nEvaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.2f}")
        logging.info("Hybrid recommendation system testing completed successfully.")
    except Exception as e:
        logging.critical("Interactive testing failed: %s", e, exc_info=True)
