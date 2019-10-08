import nltk
import random
import string, sys

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

from sklearn.metrics import classification_report

import pandas as pd
from sqlalchemy import create_engine

import re
import pickle 

from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier
from sklearn.pipeline import Pipeline

# import libraries
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    # load data from database
    '''
    Load the data with database file path
    Return : X and Y for validation and testing data, and category column values
    '''
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM Messages', engine)
    #df = pd.read_sql('SELECT * FROM disaster_response', engine)
 
    X = df['message']
    Y = df.drop(['id', 'message', 'original', 'genre'], axis = 1)
    categories = Y.columns.values
    return X,Y, categories


def tokenize(text):
    '''
    Function : Tokenize text, 
    Remove punctuation, stopwords and use lemmatizer and stemmer
    Return: List of tokens
    '''
    
    text = text.translate(str.maketrans('', '', string.punctuation)).lower()
    
    clean_tokens = []
    
     # Tokenize words
    words = word_tokenize(text)
    stop_words = nltk.corpus.stopwords.words("english")
    tokens = [w for w in words if w not in stopwords.words('english')]
    # instantiate lemmatizer
    lemmatizer = WordNetLemmatizer()
    
    # instantiate stemmer
    stemmer = PorterStemmer()
    
    clean_tokens = []
    for tok in tokens:
        # lemmtize token using noun as part of speech
        clean_tok = lemmatizer.lemmatize(tok)
        # lemmtize token using verb as part of speech
        clean_tok = lemmatizer.lemmatize(tok, pos='v')
        # stem token
        clean_tok = stemmer.stem(clean_tok)
        # strip whitespace and append clean token to array
        clean_tokens.append(clean_tok.strip())
        
    return clean_tokens


def build_model():
    '''
    Function for pipeline preparation and GridSearch
    Return: model with parameters
    '''
    pipeline = Pipeline([
    ('tfidf', TfidfVectorizer(tokenizer=tokenize)),
    ('clf', MultiOutputClassifier(RandomForestClassifier(n_estimators=10)))])


    parameters =  {'tfidf__ngram_range': ((1, 1), (1, 2)),
                'tfidf__max_df': (0.6, 1.2),
             'clf__estimator__n_estimators': [80, 120], 
              'clf__estimator__min_samples_split': [2, 6]} 
    cv = GridSearchCV(pipeline, param_grid=parameters)

    return pipeline

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Function for predicting Xtest with model and print classification report with Y_test
    '''
    y_pred = model.predict(X_test)
    #print(classification_report(Y_test, y_pred, target_names=category_names))
    for i, col in enumerate(Y_test):
        print(col)
        print(classification_report(Y_test[col], y_pred[:, i]))


def save_model(model, model_filepath):
    '''
    Function for saving model with given filepath
    '''
    Function
     with open(model_filepath, 'wb') as file:  
         pickle.dump(model, file)


def main():
    '''
    There are two arguments, filepath and path for saving the model needed to 
    start the application
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        
        #split test and train data
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