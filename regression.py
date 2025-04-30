# regression.py
# pip install pandas scikit-learn nltk gensim xgboost matplotlib

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import nltk
import gensim
from gensim.models import Word2Vec

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import xgboost as xgb

from nltk.corpus import stopwords

# Download stopwords
nltk.download('stopwords')

# Load the data
movies_df = pd.read_csv('/mnt/data/5kmovies_preprocssed.csv')

# Modeling
models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
    'XGBoost': xgb.XGBRegressor(n_estimators=100, objective='reg:squarederror', random_state=42)
}

# 6. Training and Evaluating
def get_rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

results = {}

vectorizers = {
    'Count+PCA': (X_train_count_pca, X_test_count_pca),
    'TFIDF+PCA': (X_train_tfidf_pca, X_test_tfidf_pca),
    'Word2Vec': (X_train_w2v, X_test_w2v)
}

for vec_name, (Xtr, Xte) in vectorizers.items():
    for model_name, model in models.items():
        model.fit(Xtr, y_train)
        preds = model.predict(Xte)
        rmse = get_rmse(y_test, preds)
        key = f"{model_name} ({vec_name})"
        results[key] = rmse
        print(f"{key}: RMSE = {rmse:.4f}")

# 7. Plotting Results
plt.figure(figsize=(15, 7))
plt.barh(list(results.keys()), list(results.values()))
plt.xlabel("Root Mean Squared Error (Lower is better)")
plt.title("Comparison of Models and Vectorizations on Movie Reviews")
plt.grid(True)
plt.tight_layout()
plt.show()
