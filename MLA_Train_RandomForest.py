'''
File: MLA_Train_RandomForest.py
code Cited from MLA Train.py file
vectorization code for movies1.csv
Purpose:
    This script performs machine learning analysis on movie review data using multiple text vectorization techniques
    including Bag-of-Words (BoW), TF-IDF, Word2Vec (spaCy and Google News vectors), TextBlob sentiment scores, and
    Latent Dirichlet Allocation (LDA). It uses Random Forest regression to model and predict movie ratings based on
    these vector representations. Model performance is evaluated using Root Mean Squared Error (RMSE) and R^2 score.

    The script loads pre-processed review and vector data from movie_preprocessed.csv, applies cross-validated grid search
    over a defined hyperparameter space, and visualizes actual vs predicted performance for each model.
    It also supports exporting model performance summaries to a CSV file.

Modules Used:
    - pandas, numpy, matplotlib, sklearn (feature extraction, model selection, regression, metrics)
    - nltk, gensim, spacy, textblob (for NLP and sentiment analysis)
    - joblib (optional model saving), ast, scipy.stats (statistics)
    - RandomForestRegressor (used as the primary regression model)
    - GridSearchCV for hyperparameter optimization

Key Functions:
    - mla_model(X: np.ndarray, y: List[float], modelname: str) -> Tuple[float, float, sklearn.Pipeline]:
        Trains a Random Forest regressor using a pipeline and cross-validation on the provided feature matrix X
        and target ratings y. Returns RMSE, R^2, and the best estimator.

Input:
    - "5kmovies_preprocssed.csv": Contains cleaned text reviews and movie ratings.
    - "movie_nlp_analysis.csv": Contains averaged vectorized text features and sentiment scores for each review.

Output:
    - Scatter plots of actual vs predicted ratings per vectorization method.
    - "mla_train_analysis_RandomForest.csv": CSV file with RMSE and R^2 scores for each method.

Note:
    Code and logic adapted from "MLA_Train.py" and "Vectorize.py" as part of a broader NLP movie review analysis project.
    Extensive grid search may be computationally expensive due to Random Forest tuning across multiple parameter combinations.

'''
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from io import StringIO
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import gensim.downloader as api
from gensim import corpora, models
from gensim.models import KeyedVectors
import spacy
import joblib
import ast
from scipy.stats import pearsonr
from gensim.models import CoherenceModel
from sklearn.preprocessing import StandardScaler
from textblob import TextBlob
from sklearn.model_selection import cross_val_score, train_test_split, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from yapftests.yapf_test_helper import YAPFTest
from sklearn.ensemble import RandomForestRegressor
nlp = spacy.load('en_core_web_sm')

print(f"\n\n ****Newer MLA Model***\n\n")
def mla_model(X, y, modelname):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([('scaler', StandardScaler()), ('select', SelectKBest(f_regression, k="all")),('rf', RandomForestRegressor(random_state=42))]) 
    grid = {
        'rf__n_estimators': [100,200,300,400,500],
        'rf__max_depth': [None, 10, 20, 30, 40],
        'rf__min_samples_split': [2, 5, 10],
        'rf__min_samples_leaf': [1, 2, 4],
        'rf__max_features': ['auto', 'sqrt', 'log2'],
        'rf__bootstrap': [True, False],
    }
    grids = GridSearchCV(pipe, grid, scoring='neg_mean_squared_error', cv=211, verbose=2)
    grids.fit(X_train,y_train)

    y_predicted = grids.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    r2 = r2_score(y_test, y_predicted)
    print(f"\n{modelname} Regression Results:")
    print(f"Best Parameters: {grids.best_params_}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R2 Score: {r2:.4f}")
    plt.figure(figsize=(8,6))
    plt.scatter(y_test, y_predicted, color='steelblue', alpha=0.6, edgecolors='k')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red',linestyle='--', linewidth=2)
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title(f"{modelname} Regression: Actual vs Predicted")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig(f"{modelname.replace(' ', '_')}_regression_plot_RandomForest.png")

    return rmse, r2, grids.best_estimator_

df = pd.read_csv("5kmovies_preprocssed.csv")
docs = df["cleaned_reviews"].tolist()
ratings = df["Rating"].tolist()

result_df = pd.read_csv("movie_nlp_analysis.csv")

results = []
for model_name, column_name in [
    ("BoW Avg", "BoW Vector (avg)"),
    ("TF-IDF Avg", "TF-IDF Vector (avg)"),
    ("Word2Vec Avg", "Word2Vec Vector (avg)"),
    ("Google Word2Vec Avg", "Word2Vec Vector Larger Google Model (avg)"),
    ("TextBlob Sentiment", "Sentiment Scores")
]:
    rmse, r2, best_model = mla_model(result_df[column_name].values.reshape(-1,1), ratings, model_name)
    results.append([model_name, rmse, r2])
    #joblib.dump(best_model,f"{model_name}_model.pkl")
## MLA For LDA Technique

## ** Imported from Vectorize.py @self Citation ** ##
tokenization = [doc.split() for doc in docs]
dictionary = corpora.Dictionary(tokenization)
corpus = [dictionary.doc2bow(text) for text in tokenization]

lda_model = models.LdaModel(corpus, num_topics=15, id2word=dictionary, passes=40)
lda_topics = [lda_model.get_document_topics(bow) for bow in corpus]
all_topic_distributions = [lda_model.get_document_topics(corpus[i], minimum_probability=0.0) for i in
                           range(len(corpus))]
topic_matrix = np.array([[prob for _, prob in doc] for doc in all_topic_distributions])

rmse, r2, best_model = mla_model(topic_matrix, ratings, "LDA Topic Distribution")
results.append(["LDA Topic Distribution", rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])
results_df.to_csv("mla_train_analysis_RandomForest.csv", index=False)

#oblib.dump(best_model, f"LDA_Topic_Model.pkl")
