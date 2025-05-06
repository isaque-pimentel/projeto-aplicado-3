"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides an interactive testing environment for the recommendation system.
It allows users to compare real and predicted ratings for movies they have rated and
generates top N movie recommendations for a given user.

Functions:
    get_top_n_recommendations(algo, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int, n: int = 10) -> list:
        Generates the top N recommendations for a given user.

    compare_real_and_predicted_ratings(algo, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int) -> pd.DataFrame:
        Compares real ratings with predicted ratings for movies the user has already rated.

    main():
        The entry point of the script. Loads the trained model and data, validates the data,
        and provides an interactive interface for testing recommendations.
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
    get_movie_details,
)


def get_top_n_recommendations(
    algo, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int, n: int = 10
):
    """
    Generates the top N recommendations for a given user.

    :param algo: The trained Surprise model.
    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing movie information.
    :param user_id: The ID of the user for whom to generate recommendations.
    :param n: The number of recommendations to generate.
    :return: A list of top N recommended movies with details.
    """
    logging.info("Generating top %d recommendations for user %d", n, user_id)

    # Get all movie IDs
    all_movie_ids = movies_df["MovieID"].unique()

    # Get the list of movies the user has already rated
    rated_movie_ids = ratings_df[ratings_df["UserID"] == user_id]["MovieID"].unique()

    # Predict ratings for all movies the user has not rated
    recommendations = []
    for movie_id in all_movie_ids:
        if movie_id not in rated_movie_ids:
            pred = algo.predict(user_id, movie_id)
            recommendations.append((movie_id, pred.est))

    # Sort recommendations by predicted rating
    recommendations.sort(key=lambda x: x[1], reverse=True)

    # Get details for the top N recommendations
    top_n = recommendations[:n]
    top_n_details = [
        {
            "MovieID": movie_id,
            "PredictedRating": predicted_rating,
            **get_movie_details(movie_id, movies_df),
        }
        for movie_id, predicted_rating in top_n
    ]

    return top_n_details


def compare_real_and_predicted_ratings(
    algo, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, user_id: int
):
    """
    Compares real ratings with predicted ratings for movies the user has already rated.

    :param algo: The trained Surprise model.
    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing movie information.
    :param user_id: The ID of the user.
    :return: A DataFrame with real and predicted ratings for movies the user has rated.
    """
    logging.info("Comparing real and predicted ratings for user %d", user_id)

    # Get the list of movies the user has already rated
    user_ratings = ratings_df[ratings_df["UserID"] == user_id]

    # Predict ratings for these movies
    comparisons = []
    for _, row in user_ratings.iterrows():
        movie_id = row["MovieID"]
        real_rating = row["Rating"]
        pred = algo.predict(user_id, movie_id)
        movie_details = get_movie_details(movie_id, movies_df)
        comparisons.append(
            {
                "MovieID": movie_id,
                "Title": movie_details["Title"],
                "Genres": movie_details["Genres"],
                "RealRating": real_rating,
                "PredictedRating": pred.est,
            }
        )

    # Convert to DataFrame for tabular display
    comparisons_df = pd.DataFrame(comparisons)
    return comparisons_df


if __name__ == "__main__":
    logging.info("Starting the interactive recommendation testing script.")

    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        model_path = os.path.join(project_dir, "models", "svd_model_movielens_1m.pkl")
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")

        # Load the trained model
        algo = load_model(model_path)

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

        # Compare real and predicted ratings
        comparisons_df = compare_real_and_predicted_ratings(
            algo, ratings_df, movies_df, user_id
        )
        print_table(comparisons_df, "Real vs Predicted Ratings")
        
        # Generate and display recommendations
        top_n = get_top_n_recommendations(algo, ratings_df, movies_df, user_id, n=10)

        # Generate and display recommendations
        top_n = get_top_n_recommendations(algo, ratings_df, movies_df, user_id, n=10)
        recommendations_df = pd.DataFrame(top_n)
        print_table(recommendations_df, "Top 10 Recommendations")

    except Exception as e:
        logging.critical("Interactive testing failed: %s", e, exc_info=True)
