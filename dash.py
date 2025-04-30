import dash
from dash import dcc, html
from dash.dependencies import Input, Output, State
import plotly.express as px
import pandas as pd
import numpy as np
import spacy
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression
import joblib

''' ########## install the following in terminal before running:
#pip install dash plotly pandas scikit-learn spacy gensim textblob
#python -m spacy download en_core_web_sm
'''

nlp = spacy.load('en_core_web_sm')
model = joblib.load('mla_model.pkl')  
result_df = pd.read_csv("movie_nlp_analysis.csv")

def clean_text(text):
    doc = nlp(text.lower())
    return " ".join([token.lemma_ for token in doc if not token.is_stop and not token.is_punct])

def vectorize(text, method):
    cleaned = clean_text(text)
    if method == "Word2Vec":
        doc = nlp(cleaned)
        vecs = [token.vector for token in doc if token.has_vector]
        return np.mean(vecs, axis=0).reshape(1, -1) if vecs else np.zeros((300,)).reshape(1, -1)
    elif method == "BoW":
        print("BoW vectorization requires vectorizer from training. Returning placeholder.")
        return np.zeros((1, 1))  # placeholder – can't infer BoW without the original vectorizer
    elif method == "TF-IDF":
        print("TF-IDF vectorization requires vectorizer from training. Returning placeholder.")
        return np.zeros((1, 1))  # placeholder
    elif method == "Sentiment":
        from textblob import TextBlob
        sentiment = TextBlob(cleaned).sentiment.polarity
        return np.array([[sentiment]])
    else:
        return np.zeros((1, 1))

def dominant_topic(text):
    # may need to extract LDA topics
    return "Topic 1"

# --- initialize app ---
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Movie Rating Estimator', style={'textAlign': 'center'}),
    
    html.Div([
        html.Label('Please enter your movie review:', style={'fontSize': 18}),
        dcc.Textarea(
            id='review-input',
            placeholder='Type review here...',
            style={'width': '100%', 'height': 200}
        ),
        html.Br(),
        html.Button('Predict Score', id='predict-button', n_clicks=0),
        
        html.H2(id='prediction-output', style={'marginTop': 20, 'color': 'blue'}),
        html.H3(id='topic-output', style={'marginTop': 10, 'color': 'black'})
    ], style={'width': '50%', 'margin': 'auto'}),
    
    html.Br(), html.Br(),
    
    html.Div([
        dcc.Graph(id='score-topic-graph'),
        dcc.Graph(id='length-score-graph')
    ])
])

# --- Callback Function ---
@app.callback(
    [Output('prediction-output', 'children'),
     Output('topic-output', 'children'),
     Output('score-topic-graph', 'figure'),
     Output('length-score-graph', 'figure')],
    Input('predict-button', 'n_clicks'),
    State('review-input', 'value')
)
def predict_and_update(n_clicks, review_text):
    if n_clicks == 0 or not review_text:
        return '', '', px.histogram(), px.histogram()
    
    cleaned = clean_text(review_text)
    vector = w2v_vector(cleaned).reshape(1, -1)

    predicted_score = model.predict(vector)[0]
    dominant_topic = dominant_topic(cleaned)

    # avg score per dominant topic in LDA analysis
    fig1 = px.histogram(result_df, x='dominant_topic', y='Rating', histfunc='avg',
                        title='Average Score by Dominant Topic',
                        labels={'dominant_topic': 'Topic', 'Rating': 'Avg Score'})

    # does length of review affect rating?
    result_df["review_length"] = result_df["cleaned_reviews"].apply(lambda x: len(str(x).split()))
    fig2 = px.scatter(result_df, x='review_length', y='Rating',
                      title='Review Length vs Rating',
                      labels={'review_length': 'Review Length (in words)', 'Rating': 'Score'})

    return (
        f" Predicted Score: {predicted_score:.2f}",
        f" Dominant LDA Topic: {dominant_topic}",
        fig1,
        fig2
    )

####### run app
if __name__ == '__main__':
    app.run_server(debug=True)
