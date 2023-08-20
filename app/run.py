# Import necessary libraries
import sys
import json
import pickle
import pandas as pd
from flask import Flask, render_template, request, jsonify
from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
import plotly

nltk.download(['omw-1.4', 'punkt', 'wordnet'])

app = Flask(__name__)

def tokenize(text: str):
    """Tokenize, lemmatize, and clean text data."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return cleaned_tokens

# Load data
database_path = "../data/DisasterResponse.db"
engine = create_engine("sqlite:///" + database_path)
table = pd.read_sql_table("DisasterResponse", engine)

# Load model from pickle file
model_path = "../models/classifier.pkl"
with open(model_path, 'rb') as file:
    model = pickle.load(file)

@app.route('/')
@app.route('/index')
def index():
    """Home page - this page will display the webform."""
    
    # Extract data needed for visuals
    genre_counts = table.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    # Visualization 1: Distribution of Message Genres
    graph1 = {
        'data': [
            {
                'x': genre_names,
                'y': genre_counts,
                'type': 'bar',
            },
        ],
        'layout': {
            'title': 'Distribution of Message Genres',
            'xaxis': {'title': 'Genre'},
            'yaxis': {'title': 'Count'}
        }
    }

    # Visualization 2: Distribution of Message Categories
    categories_sum = table.iloc[:, 4:].sum()
    categories_names = list(categories_sum.index)
    graph2 = {
        'data': [
            {
                'x': categories_names,
                'y': categories_sum,
                'type': 'bar',
            },
        ],
        'layout': {
            'title': 'Distribution of Message Categories',
            'xaxis': {'title': 'Category', 'tickangle': 35},
            'yaxis': {'title': 'Count'}
        }
    }

    # Create list of visuals
    graphs = [graph1, graph2]
    
    # Convert the figures to JSON
    ids = ['graph-{}'.format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

@app.route('/go')
def go():
    """Classification result page for user input message."""
    # Save user input in query
    user_input = request.args.get('query', '')
    
    # Use model to predict classification for query
    classification_labels = model.predict([user_input])[0]
    classification_results = dict(zip(table.columns[4:], classification_labels))
    
    # Render the go.html template
    return render_template(
    'go.html',
    query=user_input,
    classification_labels=classification_results,
    categories=table.columns[4:]
)

def main():
    """The main function to run the Flask application."""
    app.run(host='0.0.0.0', port=3001, debug=True)

if __name__ == '__main__':
    main()
