"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script implements the recommendation system for the MovieLens 1M dataset. It includes
functions for loading data, preprocessing temporal information, training an SVD-based
recommendation model, and evaluating its performance using metrics such as RMSE, Precision@K,
and Recall@K. The best-performing model is saved for later use.

Functions:
    load_data_from_sqlite(db_path: str) -> pd.DataFrame:
        Loads the ratings data from the SQLite database.

    preprocess_with_timestamp(ratings_df: pd.DataFrame) -> pd.DataFrame:
        Preprocesses the ratings DataFrame to include temporal information.

    precision_recall_at_k(predictions, k=10, threshold=3.5):
        Calculates Precision and Recall at K.

    save_model(algo, model_path: str) -> None:
        Saves the trained model to a file.

    train_and_evaluate_svd_with_timestamp(ratings_df: pd.DataFrame, model_path: str):
        Trains and evaluates an SVD-based recommendation system using temporal information.

    main():
        The entry point of the script. Loads the data, trains the model, evaluates it,
        and saves the best-performing model.
"""

import logging
import os
import pickle
import sqlite3
from collections import defaultdict

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader, accuracy
from surprise.model_selection import cross_validate, train_test_split

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


def precision_recall_at_k(predictions, k=10, threshold=3.5):
    """
    Calculates Precision and Recall at K.

    :param predictions: List of predictions from the Surprise library.
    :param k: Number of recommendations to consider.
    :param threshold: Rating threshold to consider a recommendation as relevant.
    :return: Precision and Recall at K.
    """
    logging.info("Calculating Precision and Recall at K.")
    user_est_true = defaultdict(list)
    for uid, _, true_r, est, _ in predictions:
        user_est_true[uid].append((est, true_r))

    precisions = dict()
    recalls = dict()

    for uid, user_ratings in user_est_true.items():
        user_ratings.sort(key=lambda x: x[0], reverse=True)
        top_k = user_ratings[:k]

        n_rel = sum((true_r >= threshold) for (_, true_r) in user_ratings)
        n_rec_k = sum((est >= threshold) for (est, _) in top_k)
        n_rel_and_rec_k = sum(
            ((true_r >= threshold) and (est >= threshold)) for (est, true_r) in top_k
        )

        precisions[uid] = n_rel_and_rec_k / n_rec_k if n_rec_k != 0 else 0
        recalls[uid] = n_rel_and_rec_k / n_rel if n_rel != 0 else 0

    avg_precision = sum(precisions.values()) / len(precisions)
    avg_recall = sum(recalls.values()) / len(recalls)
    logging.info("Precision@K: %.4f, Recall@K: %.4f", avg_precision, avg_recall)
    return avg_precision, avg_recall


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
    rmse = accuracy.rmse(predictions, verbose=True)
    logging.info("RMSE on test set: %.4f", rmse)

    # Calculate Precision and Recall at K
    precision, recall = precision_recall_at_k(predictions, k=10, threshold=3.5)
    logging.info("Precision@10: %.4f, Recall@10: %.4f", precision, recall)

    # Perform cross-validation and track the best model
    logging.info("Performing cross-validation.")
    best_rmse = float("inf")
    best_model = None

    kfolds = 5
    for fold in range(kfolds):
        trainset, testset = train_test_split(data, test_size=0.2, random_state=fold)
        algo = SVD()
        algo.fit(trainset)
        predictions = algo.test(testset)
        fold_rmse = accuracy.rmse(predictions, verbose=False)
        logging.info("Fold %d RMSE: %.4f", fold + 1, fold_rmse)

        if fold_rmse < best_rmse:
            best_rmse = fold_rmse
            best_model = algo

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

        # # Train and evaluate the recommendation system
        # train_and_evaluate_svd(ratings_df)

        logging.info("Recommendation system pipeline completed successfully.")
    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
