'''
import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.express as px
from textblob import TextBlob
import plotly.graph_objects as go
import joblib
import spacy

# Load data and models
df = pd.read_csv("movie_nlp_analysis.csv")
lda_df = pd.read_csv("5kmovies_preprocssed.csv")
results_df = pd.read_csv("mla_train_analysis.csv")

# Load trained models
bow_model = joblib.load("BoW Avg_model.pkl")
tfidf_model = joblib.load("TF-IDF Avg_model.pkl")
word2vec_model = joblib.load("Word2Vec Avg_model.pkl")
google_w2v_model = joblib.load("Google Word2Vec Avg_model.pkl")
sentiment_model = joblib.load("TextBlob Sentiment_model.pkl")
lda_model = joblib.load("LDA_Topic_Model.pkl")

nlp = spacy.load("en_core_web_sm")

def preprocess_review(text):
    doc = nlp(text.lower())
    tokens = [token.lemma_ for token in doc if token.is_alpha and not token.is_stop]
    return " ".join(tokens)

def get_features_from_review(text):
    cleaned = preprocess_review(text)
    bow_avg = np.mean([len(cleaned.split())])
    tfidf_avg = bow_avg * 0.7  # approximate
    w2v_avg = nlp(cleaned).vector.mean()
    sentiment = TextBlob(text).sentiment.polarity
    lda_topic_vector = lda_df['cleaned_reviews'].tolist() + [cleaned]
    dictionary = joblib.load("lda_dict.pkl")
    corpus = [dictionary.doc2bow(doc.split()) for doc in lda_topic_vector]
    lda_topic_matrix = np.array([[prob for _, prob in lda_model.get_document_topics(bow, minimum_probability=0.0)] for bow in corpus])
    return {
        'BoW': [[bow_avg]],
        'TF-IDF': [[tfidf_avg]],
        'Word2Vec': [[w2v_avg]],
        'Sentiment': [[sentiment]],
        'LDA': lda_topic_matrix[-1].reshape(1, -1)
    }

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
app.title = "Movie Review NLP Dashboard"

app.layout = dbc.Container([
    html.H1("📽️ Movie Review NLP Dashboard", className="text-center my-4"),
    dbc.Row([
        dbc.Col([
            html.Label("Enter a Movie Review:"),
            dcc.Textarea(id='review-input', value='', style={'width': '100%', 'height': 120}),
            dbc.Button("Predict Ratings", id="predict-btn", color="primary", className="my-2"),
        ], width=6),
        dbc.Col([
            dcc.Graph(id='rating-bar')
        ], width=6)
    ]),
    dbc.Row([
        dbc.Col([
            dcc.Graph(id='topic-distribution')
        ]),
        dbc.Col([
            dcc.Graph(id='rating-vs-sentiment')
        ])
    ]),
    html.Hr(),
    html.H5("Model Comparison Metrics"),
    dcc.Graph(figure=px.bar(results_df, x="Model", y="R2", color="Model", title="R2 Scores of NLP Models"))
], fluid=True)

@app.callback(
    [Output('rating-bar', 'figure'),
     Output('topic-distribution', 'figure'),
     Output('rating-vs-sentiment', 'figure')],
    [Input('predict-btn', 'n_clicks')],
    [State('review-input', 'value')]
)
def update_dashboard(n_clicks, review):
    if not review:
        return dash.no_update

    feats = get_features_from_review(review)
    preds = {
        'BoW': bow_model.predict(feats['BoW'])[0],
        'TF-IDF': tfidf_model.predict(feats['TF-IDF'])[0],
        'Word2Vec': word2vec_model.predict(feats['Word2Vec'])[0],
        'Sentiment': sentiment_model.predict(feats['Sentiment'])[0],
        'LDA': lda_model.predict(feats['LDA'])[0]
    }

    pred_fig = px.bar(x=list(preds.keys()), y=list(preds.values()),
                      labels={'x': 'Model', 'y': 'Predicted Rating'},
                      title="Predicted Ratings from Each Model")

    topic_fig = px.bar(x=[f"Topic {i}" for i in range(feats['LDA'].shape[1])],
                       y=feats['LDA'].flatten(),
                       labels={'x': 'Topic', 'y': 'Probability'},
                       title="LDA Topic Distribution of Review")

    sentiment_fig = px.scatter(df, x='Sentiment Scores', y='Rating',
                               title='Sentiment vs. Rating',
                               trendline="ols")

    return pred_fig, topic_fig, sentiment_fig

if __name__ == '__main__':
    app.run(debug=True)

'''
'''

import pandas as pd
import dash
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import joblib
from textblob import TextBlob
import plotly.graph_objs as go

# Load data
movie_df = pd.read_csv("movie_nlp_analysis.csv")
review_df = pd.read_csv("5kmovies_preprocssed.csv")

# Load models
bow_model = joblib.load("BoW Avg_model.pkl")
tfidf_model = joblib.load("TF-IDF Avg_model.pkl")
word2vec_model = joblib.load("Word2Vec Avg_model.pkl")
google_w2v_model = joblib.load("Google Word2Vec Avg_model.pkl")
sentiment_model = joblib.load("TextBlob Sentiment_model.pkl")
lda_model = joblib.load("LDA_Topic_Model.pkl")

# Get unique movie names
movie_options = movie_df['Movie Name'].dropna().unique()

# Initialize app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.PULSE])

app.layout = dbc.Container([
    html.H2("Movie Review Rating Predictor"),
    dcc.Dropdown(
        id='movie-dropdown',
        options=[{'label': movie, 'value': movie} for movie in movie_options],
        placeholder="Select a movie"
    ),
    html.Div(id='review-input-container'),
    html.Div(id='output-container')
])

@app.callback(
    Output('review-input-container', 'children'),
    Input('movie-dropdown', 'value')
)
def display_review_input(movie):
    if movie:
        return html.Div([
            dcc.Textarea(id='review-text', placeholder='Enter your review here...', style={'width': '100%', 'height': 150}),
            dbc.Button("Predict Ratings", id="predict-btn", color="primary", className="my-2"),
        ])
    return None

@app.callback(
    Output('output-container', 'children'),
    Input('predict-btn', 'n_clicks'),
    State('review-text', 'value'),
    State('movie-dropdown', 'value')
)
def predict_ratings(n_clicks, review, movie):
    if n_clicks and review:
        predictions = {
            'BoW': bow_model.predict([review])[0],
            'TF-IDF': tfidf_model.predict([review])[0],
            'Word2Vec': word2vec_model.predict([review])[0],
            'Google Word2Vec': google_w2v_model.predict([review])[0],
            'TextBlob Sentiment': sentiment_model.predict([review])[0],
            'LDA Topic Model': lda_model.predict([review])[0],
        }

        # Polarity analysis
        polarity = TextBlob(review).sentiment.polarity

        # Distribution plots
        full_dist = review_df['rating'].dropna()
        movie_dist = review_df[review_df['Movie Name'] == movie]['rating'].dropna()

        dist_graph = dcc.Graph(figure={
            'data': [
                go.Histogram(x=full_dist, opacity=0.6, name='All Movies'),
                go.Histogram(x=movie_dist, opacity=0.6, name=f'{movie}')
            ],
            'layout': go.Layout(title='Rating Distribution Comparison', barmode='overlay')
        })

        bar_graph = dcc.Graph(figure={
            'data': [
                go.Bar(x=list(predictions.keys()), y=list(predictions.values()), name='Predicted Ratings')
            ],
            'layout': go.Layout(title='Predicted Ratings by Model', yaxis=dict(title='Rating'))
        })

        return [
            html.H4("Prediction Results:"),
            bar_graph,
            html.H5(f"Text Polarity Score: {polarity:.2f}"),
            html.Hr(),
            html.H4("Rating Distribution"),
            dist_graph
        ]

    return None

if __name__ == '__main__':
    app.run(debug=True)
'''
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
from io import BytesIO
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import Pipeline

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

# Model loading
MODELS = {
    'BoW': 'BoW Avg_model.pkl',
    'TF-IDF': 'TF-IDF Avg_model.pkl',
    'Word2Vec': 'Word2Vec Avg_model.pkl',
    'Google_W2V': 'Google Word2Vec Avg_model.pkl',
    'Sentiment': 'TextBlob Sentiment_model.pkl',
    'LDA': 'LDA_Topic_Model.pkl'
}

models = {}
for name, path in MODELS.items():
    try:
        models[name] = joblib.load(f"models/{path}")
    except Exception as e:
        print(f"Error loading {name} model: {e}")
        models[name] = None

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
        print(image_path)
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
                                width=6,  # Adjust based on your needs
                                className="mb-4"
                            ) for _, row in analysis_df.iterrows()])
                        ], label='Model Performance'),

                        # About Tab
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
                            - Predict ratings using NLP techniques  
                            - Visualize model performance  

                            **Process**:  
                            1. Data Collection & Cleaning  
                            2. Text Preprocessing  
                            3. Feature Engineering  
                            4. Model Training & Validation  
                            5. Interactive Visualization  
                        ''', className="mb-4"),
                                html.Hr(),
                                html.H4("Technical Specifications", className="mt-4"),
                                html.Ul([
                                    html.Li("Python 3.8+"),
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

        # Generate predictions (example implementation)
        predictions = {
            'BoW': current_rating + 0.2,
            'TF-IDF': current_rating - 0.1,
            'Word2Vec': current_rating + 0.3,
            'LDA': current_rating + 0.1
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

        # Historical Ratings
        historical_fig = px.line(
            movie_data,
            y='Rating',
            title="Historical Ratings Trend",
            markers=True
        )

    return (
        rating_display,
        rating_fig,
        sentiment_fig,
        historical_fig,
        wordcloud,
        '' if review else dash.no_update
    )
# Add this temporary route to test images


if __name__ == '__main__':
    app.run(debug=True, dev_tools_props_check=False)