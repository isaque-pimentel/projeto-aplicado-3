"""
Sentiment-based recommendation core functions for HistFlix.
Authors: B Baltuilhe, I Pimentel, K Pena

This module provides functions for emotion classification and sentiment-based movie recommendation.
"""

from textblob import TextBlob
import pandas as pd

EMOTION_GENRE_MAP = {
    "happy": ["Comedy", "Adventure", "Family"],
    "sad": ["Drama", "Romance"],
    "excited": ["Action", "Sci-Fi", "Thriller"],
    "calm": ["Documentary", "History"],
    "fear": ["Thriller", "Horror"],
    "angry": ["Crime", "Action"],
    "surprise": ["Mystery", "Fantasy"],
    "neutral": ["Drama", "Documentary", "History"],
}

def classify_emotion(user_input: str) -> str:
    """
    Classifies the user's emotion based on keywords, sentiment polarity, and subjectivity.
    Returns one of the keys in EMOTION_GENRE_MAP.
    """
    sentiment = TextBlob(user_input).sentiment
    polarity = getattr(sentiment, 'polarity', 0)
    subjectivity = getattr(sentiment, 'subjectivity', 0)
    text = user_input.lower()
    emotion_keywords = {
        "happy": ["happy", "joy", "excited", "great", "awesome", "delighted", "cheerful", "content", "satisfied", "pleased", "funny", "amused"],
        "sad": ["sad", "down", "depressed", "unhappy", "melancholy", "blue", "cry", "lonely", "heartbroken", "bittersweet", "nostalgic"],
        "calm": ["calm", "relaxed", "peaceful", "serene", "chill", "tranquil", "at ease"],
        "fear": ["scared", "afraid", "fear", "terrified", "anxious", "worried", "tense", "nervous"],
        "angry": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "resentful"],
        "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected", "startled"],
        "excited": ["excited", "adrenaline", "thrill", "eager", "enthusiastic", "pumped", "curious"],
        "bored": ["bored", "dull", "uninterested", "tired", "monotonous", "meh"],
        "neutral": ["ok", "fine", "neutral", "average", "so-so", "normal"]
    }
    detected_emotions = []
    for emotion, keywords in emotion_keywords.items():
        if any(word in text for word in keywords):
            detected_emotions.append(emotion)
    if subjectivity < 0.3:
        return "neutral"
    if detected_emotions:
        if 'excited' in detected_emotions and polarity > 0.5:
            return 'excited'
        return detected_emotions[0]
    if polarity > 0.5 and subjectivity > 0.5:
        return "excited"
    elif polarity > 0.2:
        return "happy"
    elif polarity < -0.5 and subjectivity > 0.5:
        return "angry"
    elif polarity < -0.2:
        return "sad"
    elif subjectivity > 0.7:
        return "surprise"
    else:
        return "neutral"

def recommend_movies_based_on_emotion(movies_df: pd.DataFrame, emotion: str, n: int = 10) -> pd.DataFrame:
    genres = EMOTION_GENRE_MAP.get(emotion, EMOTION_GENRE_MAP["neutral"])
    def score_row(row):
        for g in genres:
            if g in row["Genres"]:
                return 2
        return 0
    movies_df = movies_df.copy()
    movies_df["EmotionScore"] = movies_df.apply(score_row, axis=1)
    recommendations = movies_df.sort_values(
        by=["EmotionScore", "Title"], ascending=[False, True]
    ).head(n)
    return recommendations
