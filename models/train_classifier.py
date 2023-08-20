# Import necessary libraries
import sys
import pandas as pd
import pickle
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier

# Download necessary NLTK data
nltk.download(['omw-1.4', 'punkt', 'wordnet'])

def retrieve_dataset(database_path: str):
    """Extract dataset from the SQLite database."""
    engine = create_engine("sqlite:///" + database_path)
    table = pd.read_sql_table("DisasterResponse", engine)
    messages = table["message"]
    labels = table.iloc[:, 4:]
    categories = labels.columns.tolist()
    return messages, labels, categories

def tokenize(text: str):
    """Tokenize, lemmatize, and clean text data."""
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    cleaned_tokens = [lemmatizer.lemmatize(token).lower().strip() for token in tokens]
    return cleaned_tokens

def setup_pipeline():
    """Construct a machine learning pipeline."""
    pipeline = Pipeline([
        ('vectorizer', CountVectorizer(tokenizer=tokenize)),
        ('transformer', TfidfTransformer()),
        ('classifier', MultiOutputClassifier(RandomForestClassifier()))
    ])
    return pipeline

def optimize_pipeline(pipeline, X_train, y_train):
    """Apply GridSearch to the given pipeline."""
    parameters = {
        'classifier__estimator__n_estimators': [50, 100],
        'classifier__estimator__min_samples_split': [2, 4]
    }
    grid_search = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1, verbose=2, cv=5)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def evaluate_performance(model, X_test, Y_test, label_names):
    """Assess model performance using test data."""
    predictions = model.predict(X_test)
    for idx, label in enumerate(label_names):
        print(f"Label: {label}\n")
        print(classification_report(Y_test.iloc[:, idx], predictions[:, idx]))

def save_trained_model(model, model_path):
    """Persist trained model to a pickle file."""
    with open(model_path, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model persisted to {model_path}")

def main_routine():
    """The main routine that organizes the complete pipeline."""
    if len(sys.argv) == 3:
        database_path, model_path = sys.argv[1:]
        print('Retrieving data...\n    DATABASE: {}'.format(database_path))
        
        messages, labels, label_names = retrieve_dataset(database_path)
        X_train, X_test, Y_train, Y_test = train_test_split(messages, labels, test_size=0.2)
        
        print('Setting up pipeline...')
        pipeline = setup_pipeline()
        
        print('Optimizing pipeline...')
        best_model = optimize_pipeline(pipeline, X_train, Y_train)
        
        print('Evaluating performance...')
        evaluate_performance(best_model, X_test, Y_test, label_names)
        
        print('Saving trained model...\n    MODEL: {}'.format(model_path))
        save_trained_model(best_model, model_path)
        
        print('Model training and persistence complete!')
    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main_routine()
