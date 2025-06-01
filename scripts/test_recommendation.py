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
from surprise import Prediction

def show_cold_start_recommendations(ratings_df, movies_df, n=10):
    """Show top-n popular movies for cold-start users."""
    movie_counts = ratings_df["MovieID"].value_counts().head(n)
    recommendations = movies_df[movies_df["MovieID"].isin(movie_counts.index)].copy()
    recommendations["NumRatings"] = recommendations["MovieID"].map(movie_counts)
    recommendations = recommendations.sort_values(by="NumRatings", ascending=False)
    print_table(recommendations, f"Top {n} Popular Movies (Cold Start)")
    print("\n--- End of evaluation for this user ---\n")

def evaluate_and_print_model(algo, label, ratings_df, movies_df, user_id, n, threshold):
    """Evaluate a model for a user and print results."""
    print(f"\n===== Results for Model: {label} =====")
    comparisons_df = compare_real_and_predicted_ratings(
        algo, ratings_df, movies_df, user_id
    )
    print_table(comparisons_df, f"Real vs Predicted Ratings ({label})")
    top_n = get_top_n_recommendations(algo, ratings_df, movies_df, user_id, n=n)
    recommendations_df = pd.DataFrame(top_n)
    print_table(recommendations_df, f"Top {n} Recommendations ({label})")
    preds = [
        Prediction(user_id, row['MovieID'], row['RealRating'], row['PredictedRating'], None)
        for _, row in comparisons_df.iterrows()
    ]
    rmse = calculate_rmse(preds)
    precision, recall = precision_recall_at_k(preds, k=n, threshold=threshold)
    print(f"RMSE: {rmse:.4f} | Precision@{n}: {precision:.2f} | Recall@{n}: {recall:.2f}")

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

        # Load all models once
        loaded_models = [load_model(path) for path in model_paths]

        while True:
            # Interactive testing
            try:
                user_id_input = input("Enter the User ID for recommendations (or -1 to exit): ")
                user_id = int(user_id_input)
            except ValueError:
                print("Invalid input. Please enter a valid integer User ID or -1 to exit.")
                continue
            if user_id == -1:
                print("Exiting interactive evaluation.")
                break
            # Check if user exists and has ratings
            if user_id not in users_df['UserID'].values or ratings_df[ratings_df['UserID'] == user_id].empty:
                show_cold_start_recommendations(ratings_df, movies_df, n=10)
                continue
            # Display user details
            user_details = get_user_details(user_id, users_df)
            print_table(pd.DataFrame([user_details]), "User Details")
            try:
                n = int(input("Enter the number of recommendations to display (default 10): ") or 10)
            except ValueError:
                n = 10
            try:
                threshold = float(input("Enter the threshold for Precision/Recall (default 3.5): ") or 3.5)
            except ValueError:
                threshold = 3.5
            for algo, label in zip(loaded_models, model_labels):
                evaluate_and_print_model(algo, label, ratings_df, movies_df, user_id, n, threshold)
            print("\n--- End of evaluation for this user ---\n")
    except Exception as e:
        logging.critical("Interactive testing failed: %s", e, exc_info=True)
