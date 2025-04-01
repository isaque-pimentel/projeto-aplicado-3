import logging
import os
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
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        ratings_df = ratings_df.astype(
            {"UserID": "int32", "MovieID": "int32", "Rating": "float32"}
        )
        logging.debug(
            "Loaded ratings table with %d rows of ratings data.", len(ratings_df)
        )

    except Exception as e:
        logging.error("Failed to load ratings data: %s", e, exc_info=True)
        raise
    finally:
        conn.close()
        logging.info("Database connection closed.")
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


def train_and_evaluate_svd(ratings_df: pd.DataFrame):
    """
    Trains and evaluates an SVD-based recommendation system.

    :param ratings_df: DataFrame containing the ratings data.
    """
    logging.info("Starting training and evaluation of SVD-based recommendation system.")

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
    logging.info("RMSE: %.4f", rmse)

    # Calculate Precision and Recall at K
    precision, recall = precision_recall_at_k(predictions, k=10, threshold=3.5)
    logging.info("Precision@10: %.4f, Recall@10: %.4f", precision, recall)

    # Perform cross-validation
    logging.info("Performing cross-validation.")
    cross_validate(algo, data, measures=["RMSE", "MAE"], cv=5, verbose=True)


if __name__ == "__main__":
    logging.info("Starting the recommendation system pipeline.")

    try:
        # Path to the SQLite database
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")

        logging.debug("Project directory: %s, DB path: %s", project_dir, db_path)

        # Load the ratings data
        ratings_df = load_data_from_sqlite(db_path)

        # Train and evaluate the recommendation system
        train_and_evaluate_svd(ratings_df)

        logging.info("Recommendation system pipeline completed successfully.")
    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
