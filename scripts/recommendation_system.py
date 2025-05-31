"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script implements the recommendation system for the MovieLens 1M dataset. It includes
functions for loading data, preprocessing temporal information, training an SVD-based
recommendation model, and evaluating its performance using metrics such as RMSE, Precision@K,
and Recall@K. The best-performing model is saved for later use.
"""

import logging
import os
import pickle
import sqlite3
from collections import defaultdict

import pandas as pd
from helpers import perform_cross_validation, precision_recall_at_k
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import train_test_split

LOG_FILE = "recommendation_system.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def load_data_from_sqlite(db_path: str) -> pd.DataFrame:
    """
    Loads the ratings data from the SQLite database.

    :param db_path: Path to the SQLite database file.
    :return: A DataFrame containing the ratings data.
    """
    logging.info("Loading ratings data from SQLite database at %s", db_path)

    conn = sqlite3.connect(db_path)
    try:
        # Load ratings table with appropriate data types
        ratings_df = pd.read_sql(
            "SELECT UserID, MovieID, Rating, Timestamp FROM ratings", conn
        )
        ratings_df = ratings_df.astype(
            {
                "UserID": "int32",
                "MovieID": "int32",
                "Rating": "float32",
                "Timestamp": "datetime64[ns]",
            }
        )
        logging.debug(
            "Loaded ratings table with %d rows of ratings data, including Timestamp.",
            len(ratings_df),
        )

    except Exception as e:
        logging.error("Failed to load ratings data: %s", e, exc_info=True)
        raise
    finally:
        conn.close()
        logging.info("Database connection closed.")
    return ratings_df


def preprocess_with_timestamp(ratings_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the ratings DataFrame to include temporal information.

    :param ratings_df: DataFrame containing the ratings data with a Timestamp column.
    :return: Preprocessed DataFrame with an additional 'Recency' feature.
    """
    logging.info("Preprocessing ratings data to include temporal information.")

    # Convert Timestamp to datetime if not already done
    if not pd.api.types.is_datetime64_any_dtype(ratings_df["Timestamp"]):
        ratings_df["Timestamp"] = pd.to_datetime(ratings_df["Timestamp"], unit="s")

    # Calculate the recency of each rating (days since the earliest rating)
    earliest_timestamp = ratings_df["Timestamp"].min()
    ratings_df["Recency"] = (ratings_df["Timestamp"] - earliest_timestamp).dt.days

    logging.debug(
        "Added 'Recency' feature to ratings DataFrame. Earliest timestamp: %s",
        earliest_timestamp,
    )
    return ratings_df


def save_model(algo, model_path: str) -> None:
    """
    Saves the trained model to a file.

    :param algo: The trained Surprise model.
    :param model_path: Path to save the model file.
    """
    logging.info("Saving the trained model to %s", model_path)
    try:
        with open(model_path, "wb") as model_file:
            pickle.dump(algo, model_file)
        logging.info("Model saved successfully.")
    except Exception as e:
        logging.error("Failed to save the model: %s", e, exc_info=True)
        raise


def train_and_evaluate_svd_with_timestamp(ratings_df: pd.DataFrame, model_path: str):
    """
    Trains and evaluates an SVD-based recommendation system using temporal information.

    :param ratings_df: DataFrame containing the ratings data with a Timestamp column.
    :param model_path: Path to save the trained model.
    """
    logging.info(
        "Starting training and evaluation of SVD-based recommendation system with temporal information."
    )

    # Preprocess the data to include temporal information
    ratings_df = preprocess_with_timestamp(ratings_df)

    # Prepare the data for Surprise
    reader = Reader(
        rating_scale=(ratings_df["Rating"].min(), ratings_df["Rating"].max())
    )
    data = Dataset.load_from_df(ratings_df[["UserID", "MovieID", "Rating"]], reader)

    # Split the data into training and testing sets
    trainset, testset = train_test_split(data, test_size=0.2, random_state=42)

    # Train the SVD model
    algo = SVD()
    algo.fit(trainset)
    logging.info("SVD model trained successfully.")

    # Evaluate the model on the test set
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions)
    logging.info("RMSE on test set: %.4f", rmse)

    # Perform cross-validation and track the best model
    best_model, best_rmse = perform_cross_validation(data, kfolds=5)
    logging.info("Best RMSE from cross-validation: %.4f", best_rmse)

    # Save the best-performing model
    if best_model:
        save_model(best_model, model_path)
        logging.info("Best-performing model saved to %s", model_path)

    # Calculate Precision and Recall at K for the best model
    if best_model:
        predictions = best_model.test(testset)
        precision, recall = precision_recall_at_k(predictions, k=10, threshold=3.5)
        logging.info("Precision@10: %.4f, Recall@10: %.4f", precision, recall)


if __name__ == "__main__":
    logging.info("Starting the recommendation system pipeline.")

    try:
        # Path to the SQLite database
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path = os.path.join(project_dir, "models", "svd_model_movielens_1m.pkl")

        logging.debug(
            "Project directory: %s, DB path: %s, Model path: %s",
            project_dir,
            db_path,
            model_path,
        )

        # Load the ratings data
        ratings_df = load_data_from_sqlite(db_path)

        # Train and evaluate the recommendation system with temporal information
        train_and_evaluate_svd_with_timestamp(ratings_df, model_path)

        logging.info("Recommendation system pipeline completed successfully.")
    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
