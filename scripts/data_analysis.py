"""
Project: HistFlix: A Personalized Recommendation System for Historical Movies and Documentaries
Authors: B Baltuilhe, I Pimentel, K Pena

This module provides functions to ....

Functions:

"""

import pandas as pd
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Tuple


def load_data(db_path: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads data from a SQLite database.

    :param db_path: Path to the SQLite database file.
    :return: A tuple containing three DataFrames: users, ratings, and movies.
    """
    conn = sqlite3.connect(db_path)
    try:
        users_df = pd.read_sql("SELECT * FROM users", conn)
        ratings_df = pd.read_sql("SELECT * FROM ratings", conn)
        movies_df = pd.read_sql("SELECT * FROM movies", conn)
    finally:
        conn.close()
    return users_df, ratings_df, movies_df


def explore_data(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
):
    """
    Explores the data by printing general information and statistics.

    :param users_df: DataFrame containing user data.
    :param ratings_df: DataFrame containing rating data.
    :param movies_df: DataFrame containing movie data.
    """
    # Display general information about the datasets
    print("\nUsers Info:\n", users_df.info(memory_usage="deep"))
    print("\nRatings Info:\n", ratings_df.info(memory_usage="deep"))
    print("\nMovies Info:\n", movies_df.info(memory_usage="deep"))

    # Display descriptive statistics for ratings
    print("\nDescriptive statistics for ratings:\n", ratings_df["Rating"].describe())

    # Display distribution of user genders
    print("\nDistribution of user genders:\n", users_df["Gender"].value_counts())

    # Display top 10 highest rated movies
    top_rated_movies = (
        ratings_df.groupby("MovieID")["Rating"]
        .mean()
        .sort_values(ascending=False)
        .head(10)
    )
    print("\nTop 10 highest rated movies:\n", top_rated_movies)

    # Optionally, merge with movies DataFrame to get movie titles
    top_rated_movies = top_rated_movies.reset_index().merge(movies_df, on="MovieID")[
        ["Title", "Rating"]
    ]
    print("\nTop 10 highest rated movies with titles:\n", top_rated_movies)


def plot_distributions(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
):
    """
    Generates plots to visualize the data distributions.

    :param users_df: DataFrame containing user data.
    :param ratings_df: DataFrame containing rating data.
    :param movies_df: DataFrame containing movie data.
    """
    plt.figure(figsize=(12, 5))
    sns.histplot(ratings_df["Rating"], bins=5, kde=True)
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()

    plt.figure(figsize=(12, 5))
    sns.countplot(x="Gender", data=users_df, palette="coolwarm")
    plt.xticks([0, 1], ["Female", "Male"])
    plt.title("Distribution of User Genders")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.show()


def perform_eda(
    users_df: pd.DataFrame, ratings_df: pd.DataFrame, movies_df: pd.DataFrame
) -> None:
    """
    Performs Exploratory Data Analysis (EDA) on the MovieLens 1M dataset.

    :param users_df: DataFrame containing user data.
    :param ratings_df: DataFrame containing rating data.
    :param movies_df: DataFrame containing movie data.
    """
    # Display general information about the datasets
    print("\nUsers Info:\n", users_df.info(memory_usage="deep"))
    print("\nRatings Info:\n", ratings_df.info(memory_usage="deep"))
    print("\nMovies Info:\n", movies_df.info(memory_usage="deep"))

    # Check for missing values
    print("\nMissing Values:\n")
    print("Users:\n", users_df.isnull().sum())
    print("Ratings:\n", ratings_df.isnull().sum())
    print("Movies:\n", movies_df.isnull().sum())

    # Display descriptive statistics for ratings
    print("\nDescriptive statistics for ratings:\n", ratings_df["Rating"].describe())

    # Distribution of ratings
    plt.figure(figsize=(10, 5))
    sns.histplot(ratings_df["Rating"], bins=5, kde=False, color="blue")
    plt.title("Distribution of Ratings")
    plt.xlabel("Rating")
    plt.ylabel("Frequency")
    plt.show()

    # Distribution of user genders
    plt.figure(figsize=(6, 4))
    sns.countplot(x="Gender", data=users_df, palette="coolwarm")
    plt.title("Distribution of User Genders")
    plt.xlabel("Gender")
    plt.ylabel("Count")
    plt.xticks([0, 1], ["Female", "Male"])
    plt.show()

    # Distribution of user ages
    plt.figure(figsize=(10, 5))
    sns.histplot(users_df["Age"], bins=10, kde=True, color="green")
    plt.title("Distribution of User Ages")
    plt.xlabel("Age")
    plt.ylabel("Frequency")
    plt.show()

    # Top 10 most rated movies
    top_rated_movies = (
        ratings_df.groupby("MovieID")["Rating"]
        .count()
        .sort_values(ascending=False)
        .head(10)
    )
    top_rated_movies = top_rated_movies.reset_index().merge(movies_df, on="MovieID")[
        ["Title", "Rating"]
    ]
    print("\nTop 10 Most Rated Movies:\n", top_rated_movies)

    # Average rating per genre
    movies_df["Genres"] = movies_df["Genres"].str.split("|")
    genre_ratings = (
        ratings_df.merge(movies_df.explode("Genres"), on="MovieID")
        .groupby("Genres")["Rating"]
        .mean()
        .sort_values(ascending=False)
    )
    print("\nAverage Rating per Genre:\n", genre_ratings)

    # Correlation analysis
    merged_df = ratings_df.merge(users_df, on="UserID")
    corr_matrix = merged_df[["Age", "Gender", "Rating"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Correlation Matrix")
    plt.show()

    # Ratings over time
    ratings_df["Timestamp"] = pd.to_datetime(ratings_df["Timestamp"], unit="s")
    ratings_over_time = ratings_df.groupby(ratings_df["Timestamp"].dt.to_period("M"))[
        "Rating"
    ].count()
    ratings_over_time.plot(figsize=(12, 6), title="Ratings Over Time", color="purple")
    plt.xlabel("Time")
    plt.ylabel("Number of Ratings")
    plt.show()


if __name__ == "__main__":
    project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    db_path = os.path.join(project_dir, "dataset", "sqlite", "movielens_1m.db")

    users, ratings, movies = load_data(db_path)
    explore_data(users, ratings, movies)
    plot_distributions(users, ratings, movies)
    perform_eda(users, ratings, movies)
