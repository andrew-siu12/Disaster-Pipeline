# Disaster Response Pipeline Project

### Installation
The following packages need to be installed for running the app:
* json
* plotly
* pandas
* flask
* numpy
* sklearn
* sqlalchemy
* nltk

### Project  Intro
This project contains code for a web app where an emergency worker can input a new message and get classification 
results in several categories, so that messages can be directed to appropriate emergency aid agencies.

###  File Descriptions
1. app
    * run.py: scripts to run the Flask web app
    * templates contains html file for the web app
2. data
    * disaster_categories.csv: categories of messages in csv format
    * disaster_messages.csv: emergency messages in csv format
    * DisasterResponse.db: Cleaned database resulting from ETL pipeline.
3. models
    * classifier.pkl: picle file containing the model output machine learning pipeline
    * train_classifier.py: ML pipeline scripts to train and tune machine learning model

### Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://127.0.0.1:5000/

### Results:
The disaster datasets provided by [Figure Eight](https://www.figure-eight.com/) are imblanaced, most of the examples are only in one category. 
As a result, the resulting model accuracy is high, but the recall and F1-score is fairly low. Logistic regression in this case perform much better than 
Random Forest Classifier. The tuned Logistic regression model have mean F1-score of 0.4319 in comparsion to only 0.2989 of tuned Random forest model.   

### Licensing, Authors, Acknowledgements
This web app was completed as part of the [Udacity Data Scientist Nanodegree](https://eu.udacity.com/course/data-scientist-nanodegree--nd025). The starter code was provided by Udacity 
and the data was provided by [Figure Eight](https://www.figure-eight.com/)  

