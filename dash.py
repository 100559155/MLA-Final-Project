app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('Movie Review Score Predictor', style={'textAlign': 'center'}),
    
    html.Div([
        html.Label('Enter your movie review:', style={'fontSize': 18}),
        dcc.Textarea(
            id='review-input',
            placeholder='Type your review here...',
            style={'width': '100%', 'height': 200}
        ),
        html.Br(),
        html.Button('Predict Score', id='predict-button', n_clicks=0),
        
        html.H2(id='prediction-output', style={'marginTop': 20, 'color': 'blue'}),
        html.H3(id='topic-output', style={'marginTop': 10, 'color': 'green'})
    ], style={'width': '50%', 'margin': 'auto'}),
    
    html.Br(), html.Br(),
    
    html.Div([
        dcc.Graph(id='score-topic-graph'),
        dcc.Graph(id='length-score-graph')
    ])
])

# --- Callback Functions ---

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
        fig1 = px.histogram()
        fig2 = px.histogram()
        return '', '', fig1, fig2
    
    cleaned = clean_text(review_text)
    vector = get_w2v_vector(cleaned).reshape(1, -1)
    prediction = model.predict(vector)[0]
    
    dominant_topic = get_dominant_topic(cleaned)
    
    # Graph 1: Distribution of scores per topic
    fig1 = px.histogram(nlp_df, x='dominant_topic', y='score', histfunc='avg',
                        title='Average Score by Dominant Topic',
                        labels={'dominant_topic': 'Topic', 'score': 'Avg Score'})
    
    # Graph 2: Review length vs score
    fig2 = px.scatter(nlp_df, x='review_length', y='score',
                      title='Review Length vs Score',
                      labels={'review_length': 'Review Length (words)', 'score': 'Score'})
    
    return (
        f" Predicted Score: {prediction:.2f}",
        f" Dominant LDA Topic: {dominant_topic}",
        fig1,
        fig2
    )

# --- Run the App ---
if __name__ == '__main__':
    app.run_server(debug=True)
