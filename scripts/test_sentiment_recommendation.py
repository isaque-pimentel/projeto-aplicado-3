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
from textblob import TextBlob
from helpers import load_model, print_table


LOG_FILE = "test_hybrid_recommendation.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def analyze_user_input(user_input: str) -> float:
    """
    Analyzes the sentiment of the user's input text.

    :param user_input: The text input from the user describing their mood and preferences.
    :return: A sentiment polarity score (-1.0 to 1.0).
    """
    logging.info("Analyzing user input sentiment.")
    sentiment_score = TextBlob(user_input).sentiment.polarity
    logging.info("User input sentiment score: %.2f", sentiment_score)
    return sentiment_score


def recommend_movies_based_on_sentiment(
    movies_df: pd.DataFrame,
    # similarity_df: pd.DataFrame,
    sentiment_score: float,
    n: int = 10,
) -> pd.DataFrame:
    """
    Recommends movies based on the user's sentiment score.

    :param movies_df: DataFrame containing movie information.
    :param similarity_df: DataFrame containing content-based similarity scores.
    :param sentiment_score: The sentiment polarity score from the user's input.
    :param n: The number of recommendations to generate.
    :return: A DataFrame containing the top N recommended movies with details.
    """
    logging.info("Recommending movies based on sentiment score %.2f.", sentiment_score)

    # # Filter movies based on sentiment score
    # if sentiment_score > 0:
    #     # Positive sentiment: Recommend movies with high sentiment scores
    #     filtered_movies = movies_df[movies_df["SentimentScore"] > 0]
    # elif sentiment_score < 0:
    #     # Negative sentiment: Recommend movies with low sentiment scores
    #     filtered_movies = movies_df[movies_df["SentimentScore"] < 0]
    # else:
    #     # Neutral sentiment: Recommend movies with average sentiment scores
    #     filtered_movies = movies_df[
    #         (movies_df["SentimentScore"] >= -0.1) & (movies_df["SentimentScore"] <= 0.1)
    #     ]

    # # Calculate the average similarity score for each movie
    # avg_similarity = similarity_df.mean(axis=1).sort_values(ascending=False)

    # # Get the top N movies
    # filtered_movies["AvgSimilarity"] = filtered_movies["MovieID"].map(avg_similarity)
    # recommendations = filtered_movies.sort_values(by="AvgSimilarity", ascending=False).head(n)

    # Filter movies based on sentiment score
    if sentiment_score > 0:
        # Positive sentiment: Recommend uplifting genres
        filtered_movies = movies_df[
            movies_df["Genres"].str.contains("Comedy|Adventure|Family", na=False)
        ]
    elif sentiment_score < 0:
        # Negative sentiment: Recommend reflective genres
        filtered_movies = movies_df[
            movies_df["Genres"].str.contains("Drama|Romance|Thriller", na=False)
        ]
    else:
        # Neutral sentiment: Recommend popular genres
        filtered_movies = movies_df[
            movies_df["Genres"].str.contains("Action|Sci-Fi|Fantasy", na=False)
        ]

    # Get the top N movies
    recommendations = filtered_movies.head(n)

    return recommendations


if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system testing script.")

    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path = os.path.join(project_dir, "models", "svd_model_movielens_1m.pkl")

        # Load the trained model
        algo = load_model(model_path)

        # Load the ratings, movies, and reviews data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()

        # Calculate content-based similarity
        # similarity_method = "tfidf"  # Change to "count" if needed
        # similarity_df = calculate_content_similarity(
        # #     movies_df, method=similarity_method
        # )

        # Interactive testing
        user_input = input("Describe your mood and what you want to watch today: ")
        sentiment_score = analyze_user_input(user_input)

        # Recommend movies based on sentiment
        recommendations = recommend_movies_based_on_sentiment(
            movies_df, sentiment_score, n=10
        )

        # Display recommendations
        print_table(recommendations, "Personalized Recommendations Based on Your Mood")

    except Exception as e:
        logging.critical("Interactive testing failed: %s", e, exc_info=True)
