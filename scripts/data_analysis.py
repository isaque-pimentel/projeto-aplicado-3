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
    level=logging.INFO,
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
        # Load users table with appropriate data types
        users_df = pd.read_sql("SELECT * FROM users", conn)
        users_df = users_df.astype(
            {
                "UserID": "int32",
                "Gender": "category",
                "Age": "int32",
                "Occupation": "int32",
            }
        )
        logging.debug("Loaded users table with %d rows.", len(users_df))

        # Load ratings table with appropriate data types
        ratings_df = pd.read_sql("SELECT * FROM ratings", conn)
        ratings_df = ratings_df.astype(
            {"UserID": "int32", "MovieID": "int32", "Rating": "float32"}
        )
        ratings_df["Timestamp"] = pd.to_datetime(ratings_df["Timestamp"])
        logging.debug("Loaded ratings table with %d rows.", len(ratings_df))

        # Load movies table with appropriate data types
        movies_df = pd.read_sql("SELECT * FROM movies", conn)
        movies_df = movies_df.astype(
            {"MovieID": "int32", "Title": "string", "Genres": "string"}
        )
        logging.debug("Loaded movies table with %d rows.", len(movies_df))

    except Exception as e:
        logging.error("Failed to load data from database: %s", e, exc_info=True)
        raise
    finally:
        conn.close()
        logging.info("Database connection closed.")

    return users_df, ratings_df, movies_df


def display_dataset_info(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    """Displays general information about the datasets."""
    logging.debug("Displaying dataset information.")
    logging.info("Dataset Information:")
    logging.info(
        "Users DataFrame: %d rows, %d columns", users_df.shape[0], users_df.shape[1]
    )
    logging.info(
        "Ratings DataFrame: %d rows, %d columns",
        ratings_df.shape[0],
        ratings_df.shape[1],
    )
    logging.info(
        "Movies DataFrame: %d rows, %d columns", movies_df.shape[0], movies_df.shape[1]
    )


def check_missing_values(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    """Checks for missing values in the datasets."""
    logging.debug("Checking for missing values.")
    logging.info("Missing Values Analysis:")
    logging.info(
        "Users DataFrame missing values:\n%s", users_df.isnull().sum().to_dict()
    )
    logging.info(
        "Ratings DataFrame missing values:\n%s", ratings_df.isnull().sum().to_dict()
    )
    logging.info(
        "Movies DataFrame missing values:\n%s", movies_df.isnull().sum().to_dict()
    )


def plot_rating_distribution(ratings_df: pd.DataFrame) -> None:
    """Plots the distribution of ratings."""
    logging.debug("Plotting distribution of ratings with percentage.")
    ratings_counts = ratings_df["Rating"].value_counts(normalize=True) * 100
    logging.info("Rating Distribution (Percentage):\n%s", ratings_counts.to_dict())
    plt.figure(figsize=(10, 5))
    sns.histplot(
        ratings_df["Rating"],
        bins=5,
        kde=False,
        color="blue",
        stat="percent",
        discrete=True,
    )
    plt.title("Distribuição de Avaliações")
    plt.xlabel("Avaliação")
    plt.ylabel("Porcentagem")
    plt.show()


def plot_user_gender_distribution(users_df: pd.DataFrame) -> None:
    """Plots the distribution of user genders."""
    logging.debug("Plotting distribution of user genders.")
    gender_counts = users_df["Gender"].value_counts()
    gender_percentages = (gender_counts / len(users_df)) * 100
    logging.info("User Gender Distribution (Counts): %s", gender_counts.to_dict())
    logging.info(
        "User Gender Distribution (Percentage): %s", gender_percentages.to_dict()
    )
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Gender", data=users_df, palette="coolwarm")
    plt.title("Distribuição de Gêneros dos Usuários")
    plt.xlabel("Gênero")
    plt.ylabel("Contagem")
    plt.xticks([0, 1], ["Feminino", "Masculino"])
    plt.show()


def plot_user_age_distribution(users_df: pd.DataFrame) -> None:
    """Plots the distribution of user ages."""
    logging.debug("Plotting distribution of user ages without KDE.")
    age_counts = users_df["Age"].value_counts().sort_index()
    age_percentages = (age_counts / len(users_df)) * 100
    logging.info("User Age Distribution (Counts): %s", age_counts.to_dict())
    logging.info("User Age Distribution (Percentage): %s", age_percentages.to_dict())
    plt.figure(figsize=(10, 5))
    sns.histplot(users_df["Age"], bins=15, kde=False, color="green")
    plt.title("Distribuição de Idades dos Usuários")
    plt.xlabel("Idade")
    plt.ylabel("Frequência")
    plt.show()


def plot_average_rating_per_genre(
    ratings_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame
) -> None:
    """Calculates and plots the average rating per genre by gender."""
    logging.debug("Calculating average rating per genre by gender.")
    movies_df["Genres"] = movies_df["Genres"].str.split("|")
    genre_ratings = (
        ratings_df.merge(movies_df.explode("Genres"), on="MovieID")
        .merge(users_df, on="UserID")
        .groupby(["Genres", "Gender"])["Rating"]
        .mean()
        .unstack()
        .sort_values(by=1, ascending=False)  # Sort by male ratings (1)
    )
    logging.info("Average Rating per Genre by Gender:\n%s", genre_ratings.to_dict())
    genre_ratings.plot(kind="bar", figsize=(12, 6), colormap="coolwarm")
    plt.title("Média de Avaliações por Gênero de Filme e Gênero de Usuário")
    plt.xlabel("Gênero de Filme")
    plt.ylabel("Média de Avaliação")
    plt.xticks(rotation=45)
    plt.legend(["Feminino", "Masculino"], title="Gênero do Usuário")
    plt.show()


def plot_ratings_over_time(ratings_df: pd.DataFrame, users_df: pd.DataFrame) -> None:
    """Analyzes and plots the evolution of ratings over time by gender."""
    logging.debug("Analyzing ratings over time by gender.")
    ratings_df["YearMonth"] = ratings_df["Timestamp"].dt.to_period("M")
    ratings_over_time = (
        ratings_df.merge(users_df, on="UserID")
        .groupby(["YearMonth", "Gender"])["Rating"]
        .count()
        .unstack()
    )
    logging.info("Ratings Over Time by Gender:\n%s", ratings_over_time.to_dict())
    ratings_over_time.plot(
        figsize=(12, 6),
        title="Avaliações ao Longo do Tempo por Gênero",
        color=["blue", "orange"],
    )
    plt.xlabel("Tempo")
    plt.ylabel("Número de Avaliações")
    plt.legend(["Feminino", "Masculino"], title="Gênero do Usuário")
    plt.show()


def plot_top_rated_movies(ratings_df: pd.DataFrame, movies_df: pd.DataFrame) -> None:
    """Identifies and plots the top 10 most rated movies."""
    logging.debug("Identifying top 10 most rated movies.")
    top_rated_movies = (
        ratings_df.groupby("MovieID")["Rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    top_rated_movies = top_rated_movies.reset_index().merge(movies_df, on="MovieID")[
        ["Title", "Rating"]
    ]
    logging.info("Top 10 Most Rated Movies:\n%s", top_rated_movies)


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
        display_dataset_info(users_df, ratings_df, movies_df)
        check_missing_values(users_df, ratings_df, movies_df)
        plot_rating_distribution(ratings_df)
        plot_user_gender_distribution(users_df)
        plot_user_age_distribution(users_df)
        plot_top_rated_movies(ratings_df, movies_df)
        plot_average_rating_per_genre(ratings_df, movies_df, users_df)
        plot_ratings_over_time(ratings_df, users_df)

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
