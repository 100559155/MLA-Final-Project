'''
Movie Rating Analysis Dashboard

This script is used to implement a dashboard for analyzing movie ratings, review sentiment, 
and model performance. The dashboard allows users to:
    - View a comparison of actual vs predicted movie ratings.
    - Analyze sentiment in movie reviews.
    - Explore historical ratings data and view word clouds for movie reviews.
    - Evaluate the performance of different regression models (BoW, TF-IDF, Word2Vec, etc.).

Steps:
    - Load pre-trained models for rating prediction, sentiment analysis, and topic modeling.
    - Preprocess review text using NLP techniques (lemmatization, stop-word removal, etc.).
    - Display interactive plots, including sentiment analysis and regression model performance.
    - Provide a word cloud for visualizing the most frequent terms in movie reviews.
    - Allow users to select a movie and input their own review for sentiment and rating prediction.

Modules Used:
    - pandas, numpy, matplotlib, plotly (for data handling and visualization)
    - dash, dash_bootstrap_components (for building the web interface)
    - joblib (for loading pre-trained models)
    - spacy (for NLP text processing)
    - textblob (for sentiment analysis)
    - gensim (for topic modeling)
    - wordcloud (for generating word clouds)
    - sklearn (for feature extraction, regression models, and metrics)
    - plotly.graph_objs (for interactive plots)

Key Functions:
    - preprocess(text: str) -> str:
        Preprocesses the input text (removes punctuation, lemmatizes, and removes stop-words).

    - create_model_card(model_row: pd.Series) -> dbc.Card:
        Generates a visual card displaying the performance metrics (RMSE, R²) and regression plot of a model.

    - generate_wordcloud(text: str) -> str:
        Generates a word cloud image from the provided text and returns it as a base64 encoded PNG.

    - get_sentiment_label(polarity: float) -> Tuple[str, str]:
        Classifies the sentiment of the text based on polarity score and returns sentiment label and color.

    - update_analysis(n_clicks: int, movie: str, review: str) -> Tuple:
        Callback function that updates the dashboard with rating predictions, sentiment analysis,
        and visualizations based on user input.

Input:
    - "5kmovies_preprocessed.csv": Contains cleaned movie review data.
    - "5kmovies.csv": Contains original movie data.
    - "mla_train_analysis.csv": Contains precomputed performance metrics for different models.
    - Pre-trained models: Ridge regression models for BoW, TF-IDF, Word2Vec, etc.

Output:
    - Interactive dashboard displaying movie rating comparisons, sentiment analysis, and historical ratings.
    - "mla_train_analysis.csv": CSV file with RMSE and R² scores for each model.

'''

# -------------------------------------------
# Import necessary libraries
# --

import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import joblib
import base64
import os
import plotly.graph_objs as go
import plotly.express as px
import re
import spacy
import numpy as np
from textblob import TextBlob
import gensim.downloader as api
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import joblib
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
import gensim.downloader as api
from gensim import corpora, models
from gensim.models import KeyedVectors
import spacy
from scipy.stats import pearsonr
from gensim.models import CoherenceModel
from textblob import TextBlob
bow_model = joblib.load("BoW Avg_model.pkl")
tfidf_model = joblib.load("TF-IDF Avg_model.pkl")
word2vec_model = joblib.load("Word2Vec Avg_model.pkl")
google_w2v_model = joblib.load("Google Word2Vec Avg_model.pkl")
sentiment_model = joblib.load("TextBlob Sentiment_model.pkl")
ldas_model = joblib.load("LDA_Topic_Model.pkl")
# Initialize the app
app = dash.Dash(__name__,
                external_stylesheets=[dbc.themes.LUX],
                meta_tags=[{'name': 'viewport',
                            'content': 'width=device-width, initial-scale=1.0'}])
server = app.server

# Load data and models
try:
    df = pd.read_csv("5kmovies_preprocssed.csv")
    kd = pd.read_csv("5kmovies.csv")
    analysis_df = pd.read_csv("mla_train_analysis.csv")
    nlp = spacy.load('en_core_web_sm')
except Exception as e:
    raise SystemExit(f"Initialization error: {e}")

# Color scheme and constants
COLORS = {
    'background': '#f8f9fa',
    'text': '#2c3e50',
    'accent': '#3498db',
    'positive': '#2ca02c',
    'negative': '#d62728',
    'neutral': '#7f7f7f'
}


# Helper functions
def preprocess(text):
    text = re.sub(r'[^\w\s]', '', text.lower())
    doc = nlp(text)
    return ' '.join([token.lemma_ for token in doc if not token.is_stop and token.is_alpha])

def create_model_card(model_row):
    """Create a model card with performance metrics and graph image"""
    try:
        # Load regression plot image
        image_path = f"assets/{model_row['Model'].replace(' ', '_')}_regression_plot.png"
        encoded_image = base64.b64encode(open(image_path, 'rb').read()).decode()
    except FileNotFoundError:
        encoded_image = ''
        print(f"Warning: Missing image for {model_row['Model']}")
    except Exception as e:
        encoded_image = ''
        print(f"Error loading image: {e}")

    return dbc.Card([
        dbc.CardHeader(model_row['Model']),
        dbc.CardBody([
            html.Div([
                dbc.Badge(f"RMSE: {model_row['RMSE']:.2f}", color="danger", className="me-1"),
                dbc.Badge(f"R²: {model_row['R2']:.2f}", color="success")
            ], className="mb-2"),
            html.Img(
                src=f'data:image/png;base64,{encoded_image}',
                style={
                    'height': '200px',
                    'width': '100%',
                    'object-fit': 'contain'
                }
            ) if encoded_image else html.Div("Image not available", className="text-muted")
        ])
    ], className="h-100 m-2")

def generate_wordcloud(text):
    wc = WordCloud(width=400, height=300, background_color='white').generate(text)
    img = BytesIO()
    wc.to_image().save(img, format='PNG')
    return 'data:image/png;base64,{}'.format(base64.b64encode(img.getvalue()).decode())


def get_sentiment_label(polarity):
    if polarity > 0.2: return ('Positive', COLORS['positive'])
    if polarity < -0.2: return ('Negative', COLORS['negative'])
    return ('Neutral', COLORS['neutral'])


# Layout
app.layout = dbc.Container(fluid=True, style={'backgroundColor': COLORS['background']}, children=[
    # Header
    dbc.Row(dbc.Col(html.H1("Movie Rating Analysis Dashboard",
                            style={'color': 'white', 'backgroundColor': COLORS['accent'], 'padding': '2rem'},
                            className="text-center"))),

            # Control Panel
            dbc.Row([
                dbc.Col([
                    html.Div([
                        dcc.Dropdown(
                            id='movie-dropdown',
                            options=[{'label': name, 'value': name} for name in df['Movie Name'].unique()],
                            placeholder="Select a Movie...",
                            className="mb-3"
                        ),
                        html.Div(id='current-rating', className="mb-3"),
                        dcc.Textarea(
                            id='review-input',
                            placeholder='Enter your review here...',
                            style={'height': '150px'},
                            className="mb-3"
                        ),
                        dbc.Button("Analyze Review", id='analyze-btn', color="primary", className="w-100")
                    ], style={'position': 'sticky', 'top': '20px'})
                ], md=3),

                # Main Content
                dbc.Col([
                    dbc.Tabs([
                        # Analysis Tab
                        dbc.Tab([
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='rating-comparison')),
                                        dbc.Col(dcc.Graph(id='sentiment-analysis'))
                            ]),
                            dbc.Row([
                                dbc.Col(dcc.Graph(id='historical-ratings')),
                                dbc.Col(html.Img(id='wordcloud'))
                            ])
                        ], label='Analysis'),

                        # Model Performance Tab
                        dbc.Tab([
                            dbc.Row([dbc.Col(dbc.Card([
                                dbc.CardHeader(model['Model']),
                                dbc.CardBody([
                                    dcc.Graph(figure=px.scatter(analysis_df, x='RMSE', y='R2',
                                                                title=f"{model['Model']} Performance")),
                                    html.Div([
                                        dbc.Badge(f"RMSE: {model['RMSE']:.2f}", color="danger"),
                                        dbc.Badge(f"R²: {model['R2']:.2f}", color="success")
                                    ], className="mt-2")
                                ])
                            ]) ) for _, model in analysis_df.iterrows()], className="g-4")
                        ], label='Model Performance'),
                        dbc.Tab([
                            dbc.Row([
                                dbc.Col(create_model_card(row),
                                width=6,
                                className="mb-4"
                            ) for _, row in analysis_df.iterrows()])
                        ], label='Model Performance'),


                        dbc.Tab([
                            html.Div([
                                html.H3("Project Overview", className="mb-4"),
                                dcc.Markdown('''
                            **Team Members**:  
                            - April Porter
                            - Filimon Keleta
                            - Lucia Enriquez 

                            **Goals**:  
                            - Analyze movie review patterns  
                            - Model Predict Comparison - predicts a single movie review effect on Rating based on pretrained model
                            - Historical Rating - shows the previous rating and current aggregated rating 
                            - Sentiment Analysis - shows word biases as positive, negative or neutral
                            - Visualize model performance  

                            **Process**:  
                            1. Data Collection & Cleaning  
                            2. Text Preprocessing for the new review 
                            3. Feature Engineering, vectorization and model training done based on loaded pickle files
                            4. Model Training & Validation  
                            5. Interactive Visualization  
                        ''', className="mb-4"),
                                html.Hr(),
                                html.H4("Technical Specifications", className="mt-4"),
                                html.Ul([
                                    html.Li("Python 3.12.2"),
                                    html.Li("Dash/Plotly Framework"),
                                    html.Li("Scikit-learn Models"),
                                    html.Li("Gensim NLP Processing")
                                ])
                            ], className="p-4")
                        ], label='About')
                    ])
                ], md=9)
            ], className="mt-4")
])

@app.callback(
    [Output('current-rating', 'children'),
     Output('rating-comparison', 'figure'),
     Output('sentiment-analysis', 'figure'),
     Output('historical-ratings', 'figure'),
     Output('wordcloud', 'src'),
     Output('review-input', 'value')],
    [Input('analyze-btn', 'n_clicks')],
    [State('movie-dropdown', 'value'),
     State('review-input', 'value')]
)



def update_analysis(n_clicks, movie, review):
    ctx = dash.callback_context
    if not ctx.triggered or not movie:
        return dash.no_update, go.Figure(), go.Figure(), go.Figure(), '', ''

    # Current Rating
    movie_data = df[df['Movie Name'] == movie]
    current_rating = movie_data['Rating'].mean()
    movie_comment = kd[kd['Movie Name'] == movie]
    Prior_comment = '.'.join(movie_comment["Reviews"].tolist())
    rating_display = html.Div([
        html.H5(f"Current Rating: {current_rating:.1f}/10",
                style={'color': COLORS['accent']}),
        html.Small(f"Previous Reviews \n {Prior_comment} ")
    ])

    # Initialize figures
    rating_fig = go.Figure()
    sentiment_fig = go.Figure()
    historical_fig = go.Figure()
    wordcloud = ''

    if review:
        # Preprocess review
        clean_review = preprocess(review)
        docs = clean_review.split()
        #This does the comparison for of Current review and how Previous review after current one is vectorized

        # Load models and vectorizers
        bow_vectorizer = joblib.load("bow_vectorizer.pkl")
        tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
        lda_vectorize = joblib.load("lda_model.pkl")
        dictionary = joblib.load("lda_dictionary.pkl")
        word2vec_vectorizer = joblib.load("spacy_model.pkl")  # spaCy model wrapper or callable
        word2vec_Google_vectorizer = KeyedVectors.load("word2vec_model.bin")

        # LDA vectorization
        bowvector = dictionary.doc2bow(docs)
        topic_distribution = lda_vectorize.get_document_topics(bowvector, minimum_probability=0.0)
        lda_vector = np.array([prob for _, prob in
                               lda_vectorize.get_document_topics(dictionary.doc2bow(clean_review.split()),
                                                             minimum_probability=0.0)]).reshape(1, -1)

        #  vector functions
        def document_vector_largermodel(doc):
            """For Google News Word2Vec"""
            tokens = doc.split()
            vectors = [word2vec_Google_vectorizer[word] for word in tokens if word in word2vec_Google_vectorizer]
            return np.mean(vectors, axis=0) if vectors else np.zeros(300)

        def document_vector(doc):
            """For spaCy GloVe"""
            return word2vec_vectorizer(doc).vector

        # Vectorizations
        bow_vector = bow_vectorizer.transform([clean_review]).toarray().mean(axis=1).reshape(1, -1)
        tfidf_vector = tfidf_vectorizer.transform([clean_review]).toarray().mean(axis=1).reshape(1, -1)
        vectorizedk_word2vec = document_vector(clean_review)
        vectorized_word2vec =np.mean(vectorizedk_word2vec).reshape(1, -1)
        vectorizedk_word2vec_google = document_vector_largermodel(clean_review)
        vectorized_word2vec_google = np.mean(vectorizedk_word2vec_google).reshape(1, -1)

        bow_pred = bow_model.predict(bow_vector)
        tfidf_pred = tfidf_model.predict(tfidf_vector)
        lda_pred = ldas_model.predict(lda_vector)
        word2vec_pred = word2vec_model.predict(vectorized_word2vec)
        google_word2vec_pred = google_w2v_model.predict(vectorized_word2vec_google)
        #total prediction based on RMSE of each model * what they returned for aggregation of review
        total_prediction = (0.0233 * bow_pred[0]) + (tfidf_pred[0] * 0.0218) + (lda_pred[0] * 0.0073) + (word2vec_pred[0] * 0.4859) + (google_word2vec_pred[0] * 0.4450 )

        # Generate predictions
        predictions = {
            'BoW': bow_pred[0],
            'TF-IDF': tfidf_pred[0],
            'Word2Vec': word2vec_pred[0],
            'LDA': lda_pred[0],
            'Word2Vec_Google' : google_word2vec_pred[0],
        }


        # Rating Comparison Figure
        rating_fig = px.bar(
            x=list(predictions.keys()),
            y=list(predictions.values()),
            labels={'x': 'Model', 'y': 'Rating'},
            title="Model Predictions Comparison"
        )

        # Sentiment Analysis
        sentiment = TextBlob(review).sentiment.polarity
        sentiment_label, color = get_sentiment_label(sentiment)
        sentiment_fig = px.pie(
            values=[abs(sentiment), 1 - abs(sentiment)],
            names=[sentiment_label, ''],
            color_discrete_sequence=[color, '#ffffff'],
            title="Sentiment Analysis"
        )

        # Word Cloud
        wordcloud = generate_wordcloud(clean_review)
        historical_data = pd.DataFrame({
            'Movie Name': [movie, movie],
            'Rating': [current_rating, total_prediction],
            'Rating Type': ['Historical Average', 'Current Prediction'],
            'Size': [15, 25]
        })

        # Historical Ratings
        historical_fig = px.scatter(
            historical_data,
            x='Rating Type',
            y='Rating',
            color='Rating Type',
            title=f"Rating Comparison for {movie}",
            labels={'Rating': 'Score (0-10)'},
            size='Size',
            size_max=20,
            hover_data=['Movie Name'],
            color_discrete_map={
                'Historical Average': '#1f77b4',
                'Current Prediction': '#ff7f0e'
            }
        )

        # Customize layout
        historical_fig.update_layout(
            xaxis_title='Rating Category',
            yaxis_range=[0, 10],
            showlegend=False
        )

        historical_fig.update_traces(
            marker=dict(line=dict(width=2, color='DarkSlateGrey')),
            selector=dict(mode='markers')
        )


    return (
        rating_display,
        rating_fig,
        sentiment_fig,
        historical_fig,
        wordcloud,
        '' if review else dash.no_update
    )



if __name__ == '__main__':
    app.run(debug=True, dev_tools_props_check=False)
