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
import pickle
import sqlite3

import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from collections import defaultdict, Counter
from surprise.model_selection import train_test_split

from scripts.helpers import (
    calculate_content_similarity,
    save_model,
    load_model,
    save_similarity_matrix,
    load_similarity_matrix,
    precompute_recommendations,
    cache_popular_items,
)

LOG_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs", "hybrid_recommendation_system.log")

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
    alpha_func=None,
) -> list:
    """
    Calculates hybrid scores for all user-movie pairs in the dataset, using a dynamic alpha per user.
    :param algo: The trained collaborative filtering model.
    :param ratings_df: DataFrame containing the ratings data.
    :param similarity_df: DataFrame containing content-based similarity scores.
    :param alpha_func: Function to compute alpha per user.
    :return: A list of tuples containing (UserID, MovieID, HybridScore).
    """
    logging.info("Calculating hybrid scores for all user-movie pairs (dynamic alpha).")
    hybrid_scores = []
    for _, row in ratings_df.iterrows():
        user_id = row["UserID"]
        movie_id = row["MovieID"]
        alpha = alpha_func(user_id, ratings_df) if alpha_func else 0.7
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


def perform_cross_validation(data, kfolds: int = 5) -> tuple:
    """
    Performs cross-validation to find the best model based on RMSE.
    :param data: Surprise Dataset object.
    :param kfolds: Number of folds for cross-validation.
    :return: The best model and its RMSE.
    """
    logging.info("Performing cross-validation with %d folds.", kfolds)
    best_rmse = float("inf")
    best_model = None
    for fold in range(kfolds):
        trainset, testset = train_test_split(data, test_size=0.2, random_state=fold)
        algo = SVD()
        algo.fit(trainset)
        predictions = algo.test(testset)
        fold_rmse = np.sqrt(
            np.mean([(true_r - est) ** 2 for (_, _, true_r, est, _) in predictions])
        )
        logging.info("Fold %d RMSE: %.4f", fold + 1, fold_rmse)
        if fold_rmse < best_rmse:
            best_rmse = fold_rmse
            best_model = algo
    logging.info("Best RMSE from cross-validation: %.4f", best_rmse)
    return best_model, best_rmse


def evaluate_hybrid_model(hybrid_scores, test_ratings_df, k=10, threshold=3.5):
    """
    Avalia o modelo híbrido usando Precision@K e Recall@K.
    :param hybrid_scores: lista de tuplas (UserID, MovieID, HybridScore)
    :param test_ratings_df: DataFrame com ratings reais do conjunto de teste
    :param k: número de recomendações
    :param threshold: limiar para considerar relevante
    """
    user_scores = defaultdict(list)
    for user_id, movie_id, score in hybrid_scores:
        user_scores[user_id].append((movie_id, score))

    precisions, recalls = [], []

    for user_id in test_ratings_df["UserID"].unique():
        relevant_items = set(
            test_ratings_df[
                (test_ratings_df["UserID"] == user_id)
                & (test_ratings_df["Rating"] >= threshold)
            ]["MovieID"]
        )
        if not relevant_items:
            continue
        ranked_items = sorted(user_scores[user_id], key=lambda x: x[1], reverse=True)
        recommended_items = [movie_id for movie_id, _ in ranked_items[:k]]
        n_rel = len(relevant_items)
        n_rec_k = len(recommended_items)
        n_rel_and_rec_k = len(set(recommended_items) & relevant_items)
        precisions.append(n_rel_and_rec_k / n_rec_k if n_rec_k else 0)
        recalls.append(n_rel_and_rec_k / n_rel if n_rel else 0)
    avg_precision = sum(precisions) / len(precisions) if precisions else 0
    avg_recall = sum(recalls) / len(recalls) if recalls else 0
    print(f"Hybrid Precision@{k}: {avg_precision:.4f}, Recall@{k}: {avg_recall:.4f}")
    return avg_precision, avg_recall


def train_and_save_hybrid_model(
    ratings_df: pd.DataFrame,
    movies_df: pd.DataFrame,
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

    # Save the collaborative filtering model
    save_model(collaborative_model, model_path)
    logging.info("Collaborative filtering model saved to %s", model_path)

    logging.info("Hybrid recommendation system training completed.")


if __name__ == "__main__":
    logging.info("Starting the hybrid recommendation system evaluation pipeline.")
    try:
        # Paths
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")
        model_path_with = os.path.join(
            project_dir, "models", "svd_movielens_1m_with_recency.pkl"
        )
        model_path_without = os.path.join(
            project_dir, "models", "svd_movielens_1m_without_recency.pkl"
        )
        # Load the ratings and movies data
        conn = sqlite3.connect(db_path)
        ratings_df = pd.read_sql("SELECT UserID, MovieID, Rating FROM ratings", conn)
        movies_df = pd.read_sql("SELECT MovieID, Title, Genres FROM movies", conn)
        conn.close()
        # Specify the similarity method (count or tfidf)
        similarity_method = "tfidf"  # or "count"
        # Load or compute similarity matrix
        sim_path = f"models/content_similarity_{similarity_method}.pkl"
        if os.path.exists(sim_path):
            similarity_df = load_similarity_matrix(sim_path)
        else:
            similarity_df = calculate_content_similarity(
                movies_df, method=similarity_method
            )
            save_similarity_matrix(similarity_df, sim_path)
        # Load SVD model (choose best)
        svd_model = load_model(model_path_with)
        users = ratings_df["UserID"].unique()
        items = ratings_df["MovieID"].unique()
        # Precompute and cache recommendations
        rec_cache_path = "models/user_recommendations.pkl"
        if os.path.exists(rec_cache_path):
            with open(rec_cache_path, "rb") as f:
                recommendations = pickle.load(f)
        else:
            recommendations = precompute_recommendations(
                svd_model,
                ratings_df,
                similarity_df,
                users,
                items,
                k=10,
                alpha=0.7,
                cache_path=rec_cache_path,
            )
        # Cache popular items for cold-start
        pop_cache_path = "models/popular_items.pkl"
        if os.path.exists(pop_cache_path):
            with open(pop_cache_path, "rb") as f:
                popular_items = pickle.load(f)
        else:
            popular_items = cache_popular_items(
                ratings_df, movies_df, k=10, cache_path=pop_cache_path
            )
        # Example: get recommendations for a user
        example_user = users[0]
        user_recs = recommendations.get(example_user, popular_items)
        logging.info(f"Recommendations for user {example_user}: {user_recs}")
        # Example: cold-start for new user
        new_user_id = max(users) + 1
        logging.info(f"Cold-start recommendations for new user: {popular_items}")
        logging.info(
            "Hybrid recommendation system evaluation pipeline completed successfully."
        )
    except Exception as e:
        logging.critical("Pipeline failed: %s", e, exc_info=True)
