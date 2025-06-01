"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides an interactive testing environment for the recommendation system.
It allows users to compare real and predicted ratings for movies they have rated and
generates top N movie recommendations for a given user.
"""

import logging
import os
import sqlite3
import pandas as pd
from helpers import (
    print_table,
    validate_data,
    load_model,
    get_user_details,
    get_top_n_recommendations,
    compare_real_and_predicted_ratings,
    precision_recall_at_k,
    calculate_rmse,
)

if __name__ == "__main__":
    logging.info("Starting the interactive recommendation testing script.")
    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_paths = [
            os.path.join(project_dir, "models", "svd_movielens_1m_with_recency.pkl"),
            os.path.join(project_dir, "models", "svd_movielens_1m_without_recency.pkl"),
        ]
        model_labels = ["With Recency", "Without Recency"]
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")

        # Load the ratings, users, and movies data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        users_df = pd.read_sql(
            "SELECT UserID, Gender, Age, Occupation FROM users", conn
        )
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()

        # Validate the data
        validate_data(users_df, ratings_df, movies_df)

        # Interactive testing
        user_id = int(input("Enter the User ID for recommendations: "))

        # Display user details
        user_details = get_user_details(user_id, users_df)
        print_table(pd.DataFrame([user_details]), "User Details")

        for model_path, label in zip(model_paths, model_labels):
            print(f"\n===== Results for Model: {label} =====")
            # Load the trained model
            algo = load_model(model_path)
            # Compare real and predicted ratings
            comparisons_df = compare_real_and_predicted_ratings(
                algo, ratings_df, movies_df, user_id
            )
            print_table(comparisons_df, f"Real vs Predicted Ratings ({label})")
            # Generate and display recommendations
            top_n = get_top_n_recommendations(algo, ratings_df, movies_df, user_id, n=10)
            recommendations_df = pd.DataFrame(top_n)
            print_table(recommendations_df, f"Top 10 Recommendations ({label})")
            # Evaluate metrics for this user/model
            preds = [algo.predict(user_id, row['MovieID']) for _, row in comparisons_df.iterrows()]
            rmse = calculate_rmse(preds)
            precision, recall = precision_recall_at_k(preds, k=10, threshold=3.5)
            print(f"RMSE: {rmse:.4f} | Precision@10: {precision:.2f} | Recall@10: {recall:.2f}")
    except Exception as e:
        logging.critical("Interactive testing failed: %s", e, exc_info=True)
