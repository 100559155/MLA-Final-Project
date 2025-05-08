"""
This file preprocesses the text from the CSV file of movie data created by movie_scraper.py
"""
import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def preprocess_text(text):
    """
    preprocess_text - clean the review data
    
    Purpose:
        This method cleans the review data by tokenizing, removing punctuation, removing stopwords,
        and removing non-alphanumeric characters.

    Parameters:
        test (string): review text

    Returns:
        Returns a list of the tokens 
    """
    doc = nlp(text.lower())
    tokens = [
        token.lemma_ for token in doc
        if token.text not in string.punctuation and
        token.text not in STOP_WORDS and
        token.is_alpha
    ]
    print(tokens)
    return " ".join(tokens)

df = pd.read_csv("5kmovies.csv")
df = df.dropna(subset=["Reviews"])
df = df[df["Reviews"].str.strip() != ""]
df["cleaned_reviews"] = df["Reviews"].apply(preprocess_text)
print(df[["Movie Name", "Rating" ,"cleaned_reviews"]].head())

df[["Movie Name", "Rating", "cleaned_reviews"]].to_csv("movies12.csv", index=False)
