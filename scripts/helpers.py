"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This script provides helper functions for the recommendation system. These functions
include utilities for printing tables, validating data, loading models, and retrieving
user and movie details.

Functions:
    print_table(df: pd.DataFrame, title: str) -> None:
        Prints a DataFrame in a tabular format using the tabulate library.

    validate_data(users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        Validates the cleaned DataFrames to ensure data integrity.

    reverse_gender_mapping(gender: int) -> str:
        Reverses the gender mapping (0 -> F, 1 -> M).

    load_model(model_path: str):
        Loads a trained model from a file.

    get_user_details(user_id: int, users_df: pd.DataFrame) -> dict:
        Retrieves user details from the users DataFrame.

    get_movie_details(movie_id: int, movies_df: pd.DataFrame) -> dict:
        Retrieves movie details from the movies DataFrame.
"""

import logging
import pickle
from collections import defaultdict

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from surprise import SVD, accuracy
from surprise.model_selection import train_test_split
from tabulate import tabulate
from textblob import TextBlob


def print_table(df: pd.DataFrame, title: str) -> None:
    """
    Prints a DataFrame in a tabular format using the tabulate library.

    :param df: The DataFrame to print.
    :param title: Title of the table.
    """
    print(f"\n{title}")
    print(tabulate(df.head(10), headers="keys", tablefmt="grid"))


def validate_data(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    """
    Validates the cleaned DataFrames to ensure data integrity.

    :param users_df: DataFrame containing user data.
    :param ratings_df: DataFrame containing rating data.
    :param movies_df: DataFrame containing movie data.
    """
    logging.info("Validating cleaned data.")

    # Check for duplicate UserIDs
    if users_df["UserID"].duplicated().any():
        raise ValueError("Duplicate UserIDs found in users DataFrame.")

    # Check for duplicate MovieIDs
    if movies_df["MovieID"].duplicated().any():
        raise ValueError("Duplicate MovieIDs found in movies DataFrame.")

    # Check for invalid ratings
    if not ratings_df["Rating"].between(0.5, 5.0).all():
        raise ValueError("Invalid ratings found in ratings DataFrame.")

    logging.info("Data validation passed successfully.")


def reverse_gender_mapping(gender: int) -> str:
    """
    Reverses the gender mapping (0 -> F, 1 -> M).

    :param gender: Mapped gender value (0 or 1).
    :return: Original gender value ('F' or 'M').
    """
    return {0: "F", 1: "M"}.get(gender, "Unknown")


def load_model(model_path: str):
    """
    Loads a trained model from a file.

    :param model_path: Path to the saved model file.
    :return: The loaded Surprise model.
    """
    logging.info("Loading the trained model from %s", model_path)
    try:
        with open(model_path, "rb") as model_file:
            algo = pickle.load(model_file)
        logging.info("Model loaded successfully.")
        return algo
    except Exception as e:
        logging.error("Failed to load the model: %s", e, exc_info=True)
        raise


def get_user_details(user_id: int, users_df: pd.DataFrame) -> dict:
    """
    Retrieves user details from the users DataFrame.

    :param user_id: The ID of the user.
    :param users_df: DataFrame containing user information.
    :return: A dictionary with user details (Gender, Age, Occupation).
    """
    user_details = users_df[users_df["UserID"] == user_id].iloc[0].to_dict()
    return user_details


def get_movie_details(movie_id: int, movies_df: pd.DataFrame) -> dict:
    """
    Retrieves movie details from the movies DataFrame.

    :param movie_id: The ID of the movie.
    :param movies_df: DataFrame containing movie information.
    :return: A dictionary with movie details (Title, Year, Genre).
    """
    movie_details = movies_df[movies_df["MovieID"] == movie_id].iloc[0].to_dict()
    return movie_details


def calculate_rmse(predictions) -> float:
    """
    Calculates the RMSE for a given set of predictions.

    :param predictions: List of predictions from the Surprise library.
    :return: The RMSE value.
    """

    rmse = accuracy.rmse(predictions, verbose=False)
    logging.info("Calculated RMSE: %.4f", rmse)
    return rmse


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
        fold_rmse = calculate_rmse(predictions)
        logging.info("Fold %d RMSE: %.4f", fold + 1, fold_rmse)

        if fold_rmse < best_rmse:
            best_rmse = fold_rmse
            best_model = algo

    logging.info("Best RMSE from cross-validation: %.4f", best_rmse)
    return best_model, best_rmse


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

    precisions = {}
    recalls = {}

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


def calculate_content_similarity(
    movies_df: pd.DataFrame, method: str = "count"
) -> pd.DataFrame:
    """
    Calculates the cosine similarity between movies based on their genres.

    :param movies_df: DataFrame containing movie information.
    :return: A DataFrame containing the similarity scores between movies.
    """
    logging.info("Calculating content-based similarity for movies.")

    # Extract the year from the title if not already present
    if "Year" not in movies_df.columns:
        movies_df["Year"] = movies_df["Title"].str.extract(r"\((\d{4})\)").fillna("")

    # Combine Genres, Title, and Year into a single string for each movie
    movies_df["CombinedFeatures"] = (
        movies_df["Genres"].fillna("")
        + "|"
        + movies_df["Title"].fillna("")
        + "|"
        + movies_df["Year"].fillna("")
        + "|"
        + movies_df["SentimentScore"].astype(str)
    )

    # Define a custom tokenizer to split on spaces
    def custom_tokenizer(text):
        return text.split("|")

    # Choose the vectorizer based on the method
    if method == "count":
        vectorizer = CountVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
    elif method == "tfidf":
        vectorizer = TfidfVectorizer(tokenizer=custom_tokenizer, token_pattern=None)
    else:
        raise ValueError("Invalid method. Choose 'count' or 'tfidf'.")

    # Create the feature matrix
    feature_matrix = vectorizer.fit_transform(movies_df["CombinedFeatures"])

    # Calculate cosine similarity between movies
    similarity_matrix = cosine_similarity(feature_matrix)

    # Convert to DataFrame for easier handling
    similarity_df = pd.DataFrame(
        similarity_matrix, index=movies_df["MovieID"], columns=movies_df["MovieID"]
    )
    logging.info("Content-based similarity calculation completed.")
    return similarity_df


def calculate_sentiment_scores(
    movies_df: pd.DataFrame, reviews_df: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculates sentiment scores for movies based on user reviews.

    :param movies_df: DataFrame containing movie information.
    :param reviews_df: DataFrame containing user reviews with MovieID and ReviewText.
    :return: DataFrame with an additional SentimentScore column for each movie.
    """
    logging.info("Calculating sentiment scores for movies.")

    # Aggregate reviews for each movie
    reviews_agg = (
        reviews_df.groupby("MovieID")["ReviewText"].apply(" ".join).reset_index()
    )

    # Calculate sentiment scores using TextBlob
    reviews_agg["SentimentScore"] = reviews_agg["ReviewText"].apply(
        lambda text: TextBlob(text).sentiment.polarity
    )

    # Merge sentiment scores with the movies DataFrame
    movies_df = movies_df.merge(
        reviews_agg[["MovieID", "SentimentScore"]], on="MovieID", how="left"
    )
    movies_df["SentimentScore"] = movies_df["SentimentScore"].fillna(
        0
    )  # Default to neutral sentiment

    logging.info("Sentiment scores calculated successfully.")
    return movies_df
