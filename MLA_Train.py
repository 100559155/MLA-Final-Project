'''
MLA_Train.py is used for training the matrix and token we have using Ridge Regression to be utilized by our dashboard. 
Steps further described in the method 
Returns results (RMSE, R2 and best model)
Plots graph 
Saves model pkl file for further use

Modules Used:
    - pandas, numpy, matplotlib, sklearn (feature extraction, model selection, regression, metrics)
    - nltk, gensim, spacy, textblob (for NLP and sentiment analysis)
    - joblib (optional model saving), ast, scipy.stats (statistics)
    - RidgeRegression (used as the primary regression model)
    - GridSearchCV for hyperparameter optimization
Key Functions:
    - mla_model(X: np.ndarray, y: List[float], modelname: str) -> Tuple[float, float, sklearn.Pipeline]:
        Trains a Ridge regressor using a pipeline and cross-validation on the provided feature matrix X
        and target ratings y. Returns RMSE, R^2, and the best estimator.
Input:
    - "5kmovies_preprocssed.csv": Contains cleaned text reviews and movie ratings.
    - "movie_nlp_analysis.csv": Contains averaged vectorized text features and sentiment scores for each review.
Output:
    - Scatter plots of actual vs predicted ratings per vectorization method.
    - "mla_train_analysis.csv": CSV file with RMSE and R^2 scores for each method.

'''

#vectorization code for movies1.csv
import matplotlib.pyplot as plt
from pandas.core.interchange.dataframe_protocol import DataFrame
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
from io import StringIO
import numpy as np
import os
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

nlp = spacy.load('en_core_web_sm')
print(f"\n\n ****Newer MLA Model***\n\n")
def mla_model(X, y, modelname):
        """
    mla_model - Train and evaluate a machine learning regression model using Ridge regression and pipeline optimization.

    Purpose:
        This method performs regression analysis using Ridge regression on the input features "X" to predict the target variable "y".
        It employs a machine learning pipeline that includes feature scaling, feature selection, and model training.
        GridSearchCV is used for hyperparameter tuning of the Ridge regression alpha value.
        The model's performance is evaluated using Root Mean Squared Error (RMSE) and R^2 score.
        A scatter plot of actual vs predicted ratings is displayed and saved as a PNG image.
        The trained best model is returned for future use @ the dashboard, via pkl file.

    Parameters:
        X (numpy.ndarray): Feature matrix used for training, typically vectorized text features (e.g., BoW, TF-IDF, LDA, Word2Vec.).
        y (list or numpy.ndarray): Target values (ratings) corresponding to the input documents.
        modelname (str): A string identifier for the model being trained, used for labeling outputs and saving results.

    Returns:
        tuple:
            - rmse (float): Root Mean Squared Error of the predictions on the test set.
            - r2 (float): R^2 Score indicating how well the model explains the variability in the target.
            - best_estimator (sklearn.pipeline.Pipeline): The best performing model pipeline after hyperparameter tuning.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipe = Pipeline([('scaler', StandardScaler()), ('select', SelectKBest(f_regression, k="all")),('ridge', Ridge())]) #creation of a pipeline
    grid = {'ridge__alpha':[0.01, 0.1, 1.0, 10.0, 100.0]}
    grids = GridSearchCV(pipe, grid, scoring='neg_mean_squared_error', cv=5)
    grids.fit(X_train,y_train) #training 
    y_predicted = grids.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_predicted))
    r2 = r2_score(y_test, y_predicted)
    print(f"\n{modelname} Regression Results:")
    print(f"Best Alpha: {grids.best_params_['ridge__alpha']}")
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
    filename = f"{modelname.replace(' ', '_')}_regression_plot.png"
    save_path = os.path.join("assets", filename)
    plt.savefig(save_path)
    plt.close()

    return rmse, r2, grids.best_estimator_

df = pd.read_csv("5kmovies_preprocssed.csv")  #reads the csv we have of the lemmatized movie rating ie, after the preprocessing pipeline
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
    joblib.dump(best_model,f"{model_name}_model.pkl")
    
## MLA For LDA Technique

## ** Imported from Vectorize.py @self Citation ** ##
tokenization = [doc.split() for doc in docs]
dictionary = corpora.Dictionary(tokenization)
corpus = [dictionary.doc2bow(text) for text in tokenization]
'''
The above two lines and the lower three lines are directly copied from vectorize.py where we used to vectorize it via LDA since we wanted it
to reaggregate itself for further use and precise resutls. 
@self citation
et al
'''
lda_model = models.LdaModel(corpus, num_topics=15, id2word=dictionary, passes=40)
lda_topics = [lda_model.get_document_topics(bow) for bow in corpus]
all_topic_distributions = [lda_model.get_document_topics(corpus[i], minimum_probability=0.0) for i in
                           range(len(corpus))] #further training of LDA, to get precise result, over 15 topics 
topic_matrix = np.array([[prob for _, prob in doc] for doc in all_topic_distributions])

rmse, r2, best_model = mla_model(topic_matrix, ratings, "LDA Topic Distribution")
results.append(["LDA Topic Distribution", rmse, r2])

results_df = pd.DataFrame(results, columns=["Model", "RMSE", "R2"])
results_df.to_csv("mla_train_analysis.csv", index=False)

joblib.dump(best_model, f"LDA_Topic_Model.pkl")

## End ##
