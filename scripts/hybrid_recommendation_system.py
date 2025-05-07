"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script trains the hybrid recommendation system for the MovieLens 1M dataset. It includes
functions for loading data, preprocessing temporal information, training an SVD-based
recommendation model, and calculating content-based similarity. The best-performing model
is saved for later use.
"""

import logging
import os
import sqlite3

import pandas as pd
from helpers import (
    get_movie_details,
    perform_cross_validation,
    save_model,
    calculate_sentiment_scores,
    calculate_content_similarity,
)
from surprise import SVD, Dataset, Reader

LOG_FILE = "hybrid_recommendation_system.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def calculate_hybrid_scores(
    algo,
    ratings_df: pd.DataFrame,
    similarity_df: pd.DataFrame,
    alpha: float = 0.7,
) -> list:
    """
    Calculates hybrid scores for all user-movie pairs in the dataset.

    :param algo: The trained collaborative filtering model.
    :param ratings_df: DataFrame containing the ratings data.
    :param similarity_df: DataFrame containing content-based similarity scores.
    :param alpha: Weight for combining collaborative and content-based scores (0 <= alpha <= 1).
    :return: A list of tuples containing (UserID, MovieID, HybridScore).
    """
    logging.info("Calculating hybrid scores for all user-movie pairs.")

    hybrid_scores = []
    for _, row in ratings_df.iterrows():
        user_id = row["UserID"]
        movie_id = row["MovieID"]

        # Collaborative filtering score
        cf_score = algo.predict(user_id, movie_id).est

        # Content-based filtering score
        cb_score = 0
        user_rated_movies = ratings_df[ratings_df["UserID"] == user_id][
            "MovieID"
        ].unique()
        for rated_movie_id in user_rated_movies:
            cb_score += similarity_df.loc[movie_id, rated_movie_id]
        cb_score /= len(user_rated_movies) if len(user_rated_movies) > 0 else 1

        # Hybrid score (weighted combination)
        hybrid_score = alpha * cf_score + (1 - alpha) * cb_score
        hybrid_scores.append((user_id, movie_id, hybrid_score))

    logging.info("Hybrid scores calculated successfully.")
    return hybrid_scores


def train_collaborative_model(ratings_df: pd.DataFrame) -> SVD:
    """
    Trains a collaborative filtering model using the Surprise library.

    :param ratings_df: DataFrame containing the ratings data.
    :return: The trained collaborative filtering model.
    """
    logging.info("Training collaborative filtering model.")

    # Prepare the data for Surprise
    reader = Reader(
        rating_scale=(ratings_df["Rating"].min(), ratings_df["Rating"].max())
    )
    data = Dataset.load_from_df(ratings_df[["UserID", "MovieID", "Rating"]], reader)

    # Perform cross-validation and track the best model
    best_model, best_rmse = perform_cross_validation(data, kfolds=5)
    logging.info("Best RMSE from cross-validation: %.4f", best_rmse)

    return best_model


def train_and_save_hybrid_model(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
    reviews_df: pd.DataFrame,
    model_path: str,
    similarity_method: str = "count",
    alpha: float = 0.7,
):
    """
    Trains and saves the hybrid recommendation system.

    :param ratings_df: DataFrame containing the ratings data.
    :param movies_df: DataFrame containing movie information.
    :param model_path: Path to save the trained model.
    :param similarity_method: The method to use for content-based similarity ("count" or "tfidf").
    :param alpha: Weight for combining collaborative and content-based scores (0 <= alpha <= 1).
    """
    logging.info("Starting training of the hybrid recommendation system.")

    # Train the collaborative filtering model
    collaborative_model = train_collaborative_model(ratings_df)

    # # Calculate sentiment scores
    # movies_df = calculate_sentiment_scores(movies_df, reviews_df)
    
    # Calculate content-based similarity
    similarity_df = calculate_content_similarity(movies_df, method=similarity_method)

    # Calculate hybrid scores
    hybrid_scores = calculate_hybrid_scores(
        collaborative_model, ratings_df, similarity_df, alpha=alpha
    )

    # Save the collaborative filtering model
    save_model(collaborative_model, model_path)
    logging.info("Collaborative filtering model saved to %s", model_path)

    logging.info("Hybrid recommendation system training completed.")


if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system training pipeline.")

    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path = os.path.join(project_dir, "models", "svd_model_movielens_1m.pkl")

        # Load the ratings and movies data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        reviews_df = pd.read_sql("SELECT MovieID, ReviewText FROM reviews", conn)
        conn.close()

        # Specify the similarity method (count or tfidf)
        similarity_method = "tfidf"  # Change to "count" if needed

        # Train and save the hybrid model
        train_and_save_hybrid_model(
            ratings_df, movies_df, model_path, similarity_method, alpha=0.7
        )

        logging.info(
            "Hybrid recommendation system training pipeline completed successfully."
        )
    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
