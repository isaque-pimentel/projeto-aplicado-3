"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This module provides functions to perform exploratory data analysis (EDA) on the MovieLens 1M dataset.

Functions:
    load_data(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        Loads data from a SQLite database.

    perform_eda(users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
        Performs Exploratory Data Analysis (EDA) on the MovieLens 1M dataset.
"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple
import logging

LOG_FILE = "data_analysis.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(funcName)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),  # Log file
        logging.StreamHandler(),  # Console output
    ],
)


def load_data(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data from a SQLite database.

    :param db_path: Path to the SQLite database file.
    :return: A tuple containing three DataFrames: users, ratings, and movies.
    """
    logging.info("Loading data from SQLite database at %s", db_path)

    conn = sqlite3.connect(db_path)
    try:
        users_df = pd.read_sql("SELECT * FROM users", conn)
        ratings_df = pd.read_sql("SELECT * FROM ratings", conn)
        movies_df = pd.read_sql("SELECT * FROM movies", conn)
        logging.debug(
            "Loaded data: users_df=%d rows, ratings_df=%d rows, movies_df=%d rows",
            len(users_df),
            len(ratings_df),
            len(movies_df),
        )
    except Exception as e:
        logging.error("Failed to load data from database: %s", e, exc_info=True)
        raise
    finally:
        conn.close()
        logging.info("Database connection closed.")

    return users_df, ratings_df, movies_df


def perform_eda(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    """
    Performs Exploratory Data Analysis (EDA) on the MovieLens 1M dataset.

    :param users_df: DataFrame containing user data.
    :param ratings_df: DataFrame containing rating data.
    :param movies_df: DataFrame containing movie data.
    """
    logging.info("Starting Exploratory Data Analysis (EDA).")

    try:
        # Display general information about the datasets
        logging.debug("Displaying dataset information.")
        logging.info("Users DataFrame info:\n%s", users_df.info(memory_usage="deep"))
        logging.info(
            "Ratings DataFrame info:\n%s", ratings_df.info(memory_usage="deep")
        )
        logging.info("Movies DataFrame info:\n%s", movies_df.info(memory_usage="deep"))

        # Check for missing values
        logging.debug("Checking for missing values.")
        logging.info("Missing values in Users DataFrame:\n%s", users_df.isnull().sum())
        logging.info(
            "Missing values in Ratings DataFrame:\n%s", ratings_df.isnull().sum()
        )
        logging.info(
            "Missing values in Movies DataFrame:\n%s", movies_df.isnull().sum()
        )

        # Display descriptive statistics for ratings
        logging.debug("Calculating descriptive statistics for ratings.")
        logging.info(
            "Descriptive statistics for ratings:\n%s", ratings_df["Rating"].describe()
        )

        # Distribution of ratings
        logging.debug("Plotting distribution of ratings.")
        plt.figure(figsize=(10, 5))
        sns.histplot(ratings_df["Rating"], bins=5, kde=False, color="blue")
        plt.title("Distribution of Ratings")
        plt.xlabel("Rating")
        plt.ylabel("Frequency")
        plt.show()

        # Distribution of user genders
        logging.debug("Plotting distribution of user genders.")
        plt.figure(figsize=(6, 4))
        sns.countplot(x="Gender", data=users_df, palette="coolwarm")
        plt.title("Distribution of User Genders")
        plt.xlabel("Gender")
        plt.ylabel("Count")
        plt.xticks([0, 1], ["Female", "Male"])
        plt.show()

        # Distribution of user ages
        logging.debug("Plotting distribution of user ages.")
        plt.figure(figsize=(10, 5))
        sns.histplot(users_df["Age"], bins=10, kde=True, color="green")
        plt.title("Distribution of User Ages")
        plt.xlabel("Age")
        plt.ylabel("Frequency")
        plt.show()

        # Top 10 most rated movies
        logging.debug("Identifying top 10 most rated movies.")
        top_rated_movies = (
            ratings_df.groupby("MovieID")["Rating"]
            .count()
            .sort_values(ascending=False)
            .head(10)
        )
        top_rated_movies = top_rated_movies.reset_index().merge(
            movies_df, on="MovieID"
        )[["Title", "Rating"]]
        logging.info("Top 10 Most Rated Movies:\n%s", top_rated_movies)

        # Average rating per genre
        logging.debug("Calculating average rating per genre.")
        movies_df["Genres"] = movies_df["Genres"].str.split("|")
        genre_ratings = (
            ratings_df.merge(movies_df.explode("Genres"), on="MovieID")
            .groupby("Genres")["Rating"]
            .mean()
            .sort_values(ascending=False)
        )
        logging.info("Average Rating per Genre:\n%s", genre_ratings)

        # Correlation analysis
        logging.debug("Performing correlation analysis.")
        merged_df = ratings_df.merge(users_df, on="UserID")
        corr_matrix = merged_df[["Age", "Gender", "Rating"]].corr()
        plt.figure(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
        plt.title("Correlation Matrix")
        plt.show()

        # Ratings over time
        logging.debug("Analyzing ratings over time.")
        ratings_df["Timestamp"] = pd.to_datetime(ratings_df["Timestamp"], unit="s")
        ratings_over_time = ratings_df.groupby(
            ratings_df["Timestamp"].dt.to_period("M")
        )["Rating"].count()
        ratings_over_time.plot(
            figsize=(12, 6), title="Ratings Over Time", color="purple"
        )
        plt.xlabel("Time")
        plt.ylabel("Number of Ratings")
        plt.show()

        logging.info("EDA completed successfully.")
    except Exception as e:
        logging.error("An error occurred during EDA: %s", e, exc_info=True)
        raise


if __name__ == "__main__":
    logging.info("Starting the HistFlix data analysis pipeline.")

    try:
        project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")

        logging.debug("Project directory: %s, DB path: %s", project_dir, db_path)

        users, ratings, movies = load_data(db_path)
        perform_eda(users, ratings, movies)
        logging.info("Data analysis pipeline completed successfully.")
    except Exception as e:
        logging.critical("Data analysis pipeline failed: %s", e, exc_info=True)
