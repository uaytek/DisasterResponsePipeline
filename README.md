# Project Overview

In this project, a model for an API that classifies disaster messages was built analyze disaster data from Figure Eight .

In the Project Workspace, there is a data set containing real messages that were sent during disaster events. A machine learning pipeline was created to categorize these events so that the messages is to send an appropriate disaster relief agency.

A web app also included where an emergency worker can input a new message and get classification results in several categories. The web app is also displaying visualizations of the data. 

Below are a few screenshots of the web app.


# Project Components
There are three components to complete for this project.

## 1. ETL Pipeline
In a Python script, etl_pipeline.py, write a data cleaning pipeline that:

Loads the messages and categories datasets
Merges the two datasets
Cleans the data
Stores it in a SQLite database
## 2. ML Pipeline
In a Python script, train_classifier.py, write a machine learning pipeline that:

Loads data from the SQLite database
Splits the dataset into training and test sets
Builds a text processing and machine learning pipeline
Trains and tunes a model using GridSearchCV
Outputs results on the test set
Exports the final model as a pickle file

## 3. Flask Web App

The flask web app flask, html, css and javascript are used for getting classification results and visualizations of the data.

# Instructions:
Run the following commands in the project's root directory to set up your database and model.

To run ETL pipeline that cleans data and stores in database python data/etl_pipeline.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db
To run ML pipeline that trains classifier and saves python models/train_classifier.py data/DisasterResponse.db models/disaster_model.pkl
Run the following command in the app's directory to run your web app. python run.py

## Required packages:
nltk
pickle
flask
joblib
pandas
plot.ly
numpy
scikit-learn
sqlalchemy
