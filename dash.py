# Build the Dash App

# Initialize Dash app
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1('🎬 Movie Review Score Predictor', style={'textAlign': 'center'}),
    
    html.Div([
        html.Label('Enter your movie review:', style={'fontSize': 18}),
        dcc.Textarea(
            id='review-input',
            placeholder='Type your review here...',
            style={'width': '100%', 'height': 200}
        ),
        
        html.Br(),
        html.Button('Predict Score', id='predict-button', n_clicks=0),
        
        html.H2(id='prediction-output', style={'marginTop': 20, 'color': 'blue'})
    ], style={'width': '50%', 'margin': 'auto'})
])

# 3. Callback Function
@app.callback(
    Output('prediction-output', 'children'),
    Input('predict-button', 'n_clicks'),
    State('review-input', 'value')
)
def predict_review_score(n_clicks, review_text):
    if n_clicks == 0 or not review_text:
        return ''
    
    cleaned = clean_text(review_text)
    vector = get_w2v_vector(cleaned).reshape(1, -1)
    prediction = model.predict(vector)[0]
    
    return f"⭐ Predicted Score: {prediction:.2f}"

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
