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

import pickle
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
    category_names: list of str. List containing the column names of Y labels
    """
    engine = create_engine('sqlite:///Messages.db')
    df = pd.read_sql('SELECT * FROM Messages', engine)
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis=1)

    category_names = list(Y.columns.values)
    return X, Y, category_names


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


def score(y_true, y_pred):
    """
    Calculate median F1 score for multi-output labels

    :param y_true: array. Actual labels
    :param y_pred: array. Predicted labels return from models

    :returns
    f1_median: float. median f1 scores
    """
    y_true = np.array(y_true)
    f1_list = []
    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        f1_list.append(f1)
    f1_median = np.median(f1_list)
    return f1_median


def score(y_true, y_pred):
    """
    Calculate median F1 score for multi-output labels

    :param y_true: array. Actual labels
    :param y_pred: array. Predicted labels return from models

    :returns
    f1_median: float. median f1 scores
    """
    y_true = np.array(y_true)
    f1_list = []
    for i in range(y_true.shape[1]):
        f1 = f1_score(y_true[:, i], y_pred[:, i])
        f1_list.append(f1)
    f1_median = np.median(f1_list)
    return f1_median


def build_model():
    """
    To build machine learning pipeline
    :return:
    cv_log: GridSearch CV object  for logistic regression
    """
    pipeline_log = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(LogisticRegression()))
    ])

    parameters_log = {
        'vect__max_df': [1.0],
        'tfidf__use_idf': [True],
        'clf__estimator__penalty': ['l1', 'l2'],
        'clf__estimator__C': np.logspace(-5, 5, 10)}

    scorer = make_scorer(score)
    cv_log = GridSearchCV(pipeline_log, param_grid=parameters_log, scoring=scorer, verbose=5)
    return cv_log


def evaluate_model(model, X_test, Y_test, category_names):
    """
    Print evaluation metrics of the model

    :param model: sklearn models or pipelines
    :param X_test: dataframe. test feature set
    :param Y_test:  dataframe. test labels set
    :param category_names: list of str. List containing the column names of Y labels
    """
    Y_test = np.array(Y_test)
    Y_pred = model.predict(X_test)
    metrics = []

    for i in range(len(category_names)):
        accuracy = accuracy_score(Y_test[:, i], Y_pred[:, i])
        precision = precision_score(Y_test[:, i], Y_pred[:, i])
        recall = recall_score(Y_test[:, i], Y_pred[:, i])
        f1 = f1_score(Y_test[:, i], Y_pred[:, i])
        metrics.append([accuracy, precision, recall, f1])

    metrics_df = pd.DataFrame(metrics, index=category_names, columns=['Accuracy', 'Precision', 'Recall', 'F1-score'])
    print(metrics_df)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as out_file:
        pickle.dump(model.best_estimator_, out_file)


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