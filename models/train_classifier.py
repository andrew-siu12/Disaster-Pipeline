import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import recall_score, accuracy_score, precision_score, f1_score, make_scorer
from sklearn.linear_model import LogisticRegression

import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer

import warnings

warnings.filterwarnings("ignore")

def load_data(database_filepath):
    """
     Load , merge msessage and categories  dataset and seperate  into features and labels

    :param messages_filepath: str: Filepath for messages.csv file
    :param categories_filepath:: Filepath for categories.csv file
    :return:
    X: pandas dataframe. Feature dataset
    Y: pandas dataframe: Labels dataset
    """
    engine = create_engine('sqlite:///Messages.db')
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    return X, Y


def tokenize(text):
    """
    Normalize, tokenize, remove stopwords and lemmatize words

    :param ext: str. Messages for preprocessing
    :returns
    clean_tokens: list of strings. Containing tokenize words
    """
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())

    # tokenize text
    tokens = word_tokenize(text)

    # Initialize Lemmatizer
    lemma = WordNetLemmatizer()

    lemmed = [lemma.lemmatize(w) for w in tokens if w not in stopwords.words("english")]

    return lemmed


def build_model():
    pass


def evaluate_model(model, X_test, Y_test, category_names):
    pass


def save_model(model, model_filepath):
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()