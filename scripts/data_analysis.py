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

        # Graph for statistics of ratings (violin plot)
        logging.debug("Creating violin plot for rating statistics.")
        plt.figure(figsize=(8, 6))
        sns.violinplot(x="Rating", data=ratings_df, palette="muted")
        plt.title("Distribuição de Estatísticas de Avaliações")
        plt.xlabel("Avaliação")
        plt.ylabel("Densidade")
        plt.show()

        # Distribution of ratings (with narrower bars and percentage on y-axis)
        logging.debug("Plotting distribution of ratings with percentage.")
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

        # Distribution of user genders
        logging.debug("Plotting distribution of user genders.")
        plt.figure(figsize=(6, 4))
        sns.countplot(x="Gender", data=users_df, palette="coolwarm")
        plt.title("Distribuição de Gêneros dos Usuários")
        plt.xlabel("Gênero")
        plt.ylabel("Contagem")
        plt.xticks([0, 1], ["Feminino", "Masculino"])
        plt.show()

        # Distribution of user ages (narrower bars, no KDE)
        logging.debug("Plotting distribution of user ages without KDE.")
        plt.figure(figsize=(10, 5))
        sns.histplot(users_df["Age"], bins=15, kde=False, color="green")
        plt.title("Distribuição de Idades dos Usuários")
        plt.xlabel("Idade")
        plt.ylabel("Frequência")
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

        # Average rating per genre (by gender)
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
        logging.info("Average Rating per Genre by Gender:\n%s", genre_ratings)
        genre_ratings.plot(kind="bar", figsize=(12, 6), colormap="coolwarm")
        plt.title("Média de Avaliações por Gênero de Filme e Gênero de Usuário")
        plt.xlabel("Gênero de Filme")
        plt.ylabel("Média de Avaliação")
        plt.xticks(rotation=45)
        plt.legend(["Feminino", "Masculino"], title="Gênero do Usuário")
        plt.show()

        # Ratings over time (evolution by gender)
        logging.debug("Analyzing ratings over time by gender.")
        ratings_df["YearMonth"] = ratings_df["Timestamp"].dt.to_period("M")
        ratings_over_time = (
            ratings_df.merge(users_df, on="UserID")
            .groupby(["YearMonth", "Gender"])["Rating"]
            .count()
            .unstack()
        )
        ratings_over_time.plot(
            figsize=(12, 6),
            title="Avaliações ao Longo do Tempo por Gênero",
            color=["blue", "orange"],
        )
        plt.xlabel("Tempo")
        plt.ylabel("Número de Avaliações")
        plt.legend(["Feminino", "Masculino"], title="Gênero do Usuário")
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
