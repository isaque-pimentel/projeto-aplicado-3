"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides an interactive testing environment for the hybrid recommendation system.
It allows users to generate hybrid recommendations by combining collaborative filtering
and content-based filtering.
"""

import logging
import os
import sqlite3

import pandas as pd
from scripts.helpers import (
    calculate_content_similarity,
    evaluate_recommendations,
    load_model,
    print_table,
)
from hybrid_recommendation_system import calculate_hybrid_scores

LOG_FILE = "test_hybrid_recommendation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def get_top_n_content_recommendations(
    movies_df: pd.DataFrame, similarity_df: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """
    Generates the top N content-based recommendations for new users.

    :param movies_df: DataFrame containing movie information.
    :param similarity_df: DataFrame containing content-based similarity scores.
    :param n: The number of recommendations to generate.
    :return: A DataFrame containing the top N recommended movies with details.
    """
    logging.info("Generating top %d content-based recommendations for new users.", n)

    # Calculate the average similarity score for each movie
    avg_similarity = similarity_df.mean(axis=1).sort_values(ascending=False)

    # Get the top N movies
    top_n_movies = avg_similarity.head(n).index
    recommendations = movies_df[movies_df["MovieID"].isin(top_n_movies)].copy()
    recommendations["AvgSimilarity"] = recommendations["MovieID"].map(avg_similarity)

    return recommendations.sort_values(by="AvgSimilarity", ascending=False)


def get_default_recommendations(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame, n: int = 10
) -> pd.DataFrame:
    """
    Generates default recommendations based on the most popular or highly-rated movies.

    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing movie information.
    :param n: The number of recommendations to generate.
    :return: A DataFrame containing the top N default recommended movies with details.
    """
    logging.info("Generating default recommendations based on popularity.")

    # Calculate the average rating and the number of ratings for each movie
    movie_stats = ratings_df.groupby("MovieID").agg(
        AvgRating=("Rating", "mean"), NumRatings=("Rating", "count")
    )
    movie_stats = movie_stats.sort_values(
        by=["NumRatings", "AvgRating"], ascending=False
    )

    # Get the top N movies
    top_n_movies = movie_stats.head(n).index
    recommendations = movies_df[movies_df["MovieID"].isin(top_n_movies)].copy()
    recommendations = recommendations.merge(movie_stats, on="MovieID")

    return recommendations.sort_values(by=["NumRatings", "AvgRating"], ascending=False)


def get_top_n_hybrid_recommendations(
    algo,
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    user_id: int,
    similarity_df: pd.DataFrame,
    n: int = 10,
    alpha: float = 0.7,
) -> pd.DataFrame:
    """
    Generates the top N hybrid recommendations for a given user, handling cold start scenarios.

    :param algo: The trained collaborative filtering model.
    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing movie information.
    :param user_id: The ID of the user for whom to generate recommendations.
    :param similarity_df: DataFrame containing content-based similarity scores.
    :param n: The number of recommendations to generate.
    :param alpha: Weight for combining collaborative and content-based scores (0 <= alpha <= 1).
    :return: A DataFrame containing the top N recommended movies with details.
    """
    logging.info("Generating top %d hybrid recommendations for user %d.", n, user_id)

    # Check if the user has rated any movies
    user_ratings = ratings_df[ratings_df["UserID"] == user_id]
    if user_ratings.empty:
        logging.info(
            "No ratings found for user %d. Falling back to content-based recommendations.",
            user_id,
        )
        return get_top_n_content_recommendations(movies_df, similarity_df, n)

    # Calculate hybrid scores for the user
    hybrid_scores = calculate_hybrid_scores(algo, user_ratings, similarity_df, alpha)

    # Sort by hybrid score and get the top N recommendations
    hybrid_scores.sort(key=lambda x: x[2], reverse=True)
    top_n_scores = hybrid_scores[:n]

    # Get movie details for the top N recommendations
    recommendations = []
    for _, movie_id, hybrid_score in top_n_scores:
        movie_details = movies_df[movies_df["MovieID"] == movie_id].iloc[0].to_dict()
        recommendations.append(
            {
                "MovieID": movie_id,
                "Title": movie_details["Title"],
                "Genres": movie_details["Genres"],
                "HybridScore": hybrid_score,
            }
        )

    return pd.DataFrame(recommendations)


if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system testing script.")

    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path = os.path.join(project_dir, "models", "svd_model_movielens_1m.pkl")

        # Load the trained model
        algo = load_model(model_path)

        # Load the ratings and movies data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()

        # Calculate content-based similarity
        similarity_method = "tfidf"  # Change to "count" if needed
        similarity_df = calculate_content_similarity(
            movies_df, method=similarity_method
        )

        # Interactive testing
        user_id = int(input("Enter the User ID for recommendations: "))
        user_ratings = ratings_df[ratings_df["UserID"] == user_id]

        # Generate hybrid recommendations
        top_n = calculate_hybrid_scores(algo, user_ratings, similarity_df, alpha=0.7)
        recommendations = pd.DataFrame(
            [
                {
                    "MovieID": movie_id,
                    "Title": movies_df[movies_df["MovieID"] == movie_id][
                        "Title"
                    ].values[0],
                    "Genres": movies_df[movies_df["MovieID"] == movie_id][
                        "Genres"
                    ].values[0],
                    "HybridScore": score,
                }
                for _, movie_id, score in top_n
            ]
        )

        # Display recommendations
        print_table(
            recommendations, f"Top 10 Hybrid Recommendations for User {user_id}"
        )

        # Evaluate recommendations
        metrics = evaluate_recommendations(recommendations, user_ratings, n=10)
        print("\nEvaluation Metrics:")
        for metric, value in metrics.items():
            print(f"{metric}: {value:.2f}")
    except Exception as e:
        logging.critical("Interactive testing failed: %s", e, exc_info=True)
