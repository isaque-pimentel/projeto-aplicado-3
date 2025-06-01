"""
Sentiment-based recommendation core functions for HistFlix.
Authors: B Baltuilhe, I Pimentel, K Pena

This module provides functions for emotion classification and sentiment-based movie recommendation.
"""

from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
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

EMOTION_KEYWORDS = {
    "happy": ["happy", "joy", "delighted", "cheerful", "content", "satisfied", "pleased", "funny", "amused", "smile", "laugh"],
    "sad": ["sad", "down", "depressed", "unhappy", "melancholy", "blue", "cry", "lonely", "heartbroken", "bittersweet", "nostalgic"],
    "calm": ["calm", "relaxed", "peaceful", "serene", "chill", "tranquil", "at ease"],
    "fear": ["scared", "afraid", "fear", "terrified", "anxious", "worried", "tense", "nervous"],
    "angry": ["angry", "mad", "furious", "annoyed", "irritated", "frustrated", "resentful"],
    "surprise": ["surprised", "shocked", "amazed", "astonished", "unexpected", "startled"],
    "excited": ["excited", "adrenaline", "thrill", "eager", "enthusiastic", "pumped", "curious", "energetic"],
    "bored": ["bored", "dull", "uninterested", "tired", "monotonous", "meh"],
    "neutral": ["ok", "fine", "neutral", "average", "so-so", "normal"]
}

analyzer = SentimentIntensityAnalyzer()

def classify_emotion(user_input: str) -> str:
    """
    Classifies the user's emotion based on keywords, sentiment polarity, and subjectivity.
    Returns one of the keys in EMOTION_GENRE_MAP.
    If the input is too short or only whitespace, returns 'neutral'.
    """
    if not user_input or not user_input.strip() or len(user_input.strip()) < 3:
        return "neutral"
    sentiment = TextBlob(user_input).sentiment
    polarity = getattr(sentiment, 'polarity', 0)
    subjectivity = getattr(sentiment, 'subjectivity', 0)
    text = user_input.lower()
    detected_emotions = []
    for emotion, keywords in EMOTION_KEYWORDS.items():
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

def detect_emotions_with_weights(text: str):
    """
    Detects multiple emotions in the text and assigns a weight to each based on keyword frequency and VADER compound score.
    Returns a dict: emotion -> weight (normalized to sum to 1).
    """
    text_lower = text.lower()
    emotion_scores = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    # Keyword-based scoring
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                emotion_scores[emotion] += 1
    # VADER sentiment
    vader_scores = analyzer.polarity_scores(text)
    # Map VADER compound to emotion weights
    if vader_scores['compound'] > 0.5:
        emotion_scores['happy'] += vader_scores['compound']
        emotion_scores['excited'] += vader_scores['pos']
    elif vader_scores['compound'] < -0.5:
        emotion_scores['sad'] += abs(vader_scores['compound'])
        emotion_scores['angry'] += vader_scores['neg']
    elif abs(vader_scores['compound']) < 0.2:
        emotion_scores['neutral'] += 1
    # Normalize
    total = sum(emotion_scores.values())
    if total == 0:
        return {'neutral': 1.0}
    return {emo: score/total for emo, score in emotion_scores.items() if score > 0}

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

def recommend_movies_multi_emotion(movies_df: pd.DataFrame, emotion_weights: dict, n: int = 10) -> pd.DataFrame:
    """
    Recommend movies by boosting genres according to weighted emotions.
    """
    movies_df = movies_df.copy()
    def score_row(row):
        score = 0
        for emotion, weight in emotion_weights.items():
            genres = EMOTION_GENRE_MAP.get(emotion, [])
            if any(g in row["Genres"] for g in genres):
                score += 2 * weight
        return score
    movies_df["EmotionScore"] = movies_df.apply(score_row, axis=1)
    recommendations = movies_df.sort_values(
        by=["EmotionScore", "Title"], ascending=[False, True]
    ).head(n)
    return recommendations
