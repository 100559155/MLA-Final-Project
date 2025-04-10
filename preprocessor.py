import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd
nlp = spacy.load("en_core_web_sm")
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer

def preprocess_text(text):
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
