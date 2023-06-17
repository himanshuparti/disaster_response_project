import sys

import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet', 'omw-1.4'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.multioutput import MultiOutputClassifier
import pickle


def load_data(database_filepath):
    """
    load_data
   Load Data from the Database file
   
   Arguments:
       database_filepath -> Path to SQLite destination database - DisasterResponse.db)
   Output:
       X -> a dataframe containing message
       y -> a dataframe containing labels
       category_names -> List of categories name
   """
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disaster_Data', engine)
    # splitting into target and predicting variables
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names



def tokenize(text):
    """
    tokenize
    Tokenize and lemmatize the text 
    
    Arguments:
        text -> Text message which needs to be tokenized
    Output:
        token_list -> List of tokens extracted from the provided text
    """
    url_regular = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    urls = re.findall(url_regular, text)
    for url in urls:
        text = text.replace(url, "urlplaceholder")
    
    token = word_tokenize(text)
    lemma = WordNetLemmatizer()
    
    token_list = []
    for tok in token:
        clean_tok = lemma.lemmatize(tok).lower().strip()
        token_list.append(clean_tok)
    
    return token_list


def build_model():
    """
   build_model
   A Scikit ML model that process text messages and applies a classifier.
       
   """
    pipeline = Pipeline([
    ('vec_count', CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('mo_clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'mo_clf__estimator__n_estimators': [40],
        'mo_clf__estimator__min_samples_split': [3],
    
    }
    model = GridSearchCV(pipeline, param_grid=parameters, n_jobs=2, verbose=2, cv=3)
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    """
    evaluate_model
    This function prints out the model performance using classification matrix
    
    Arguments:
        model -> scikit model
        X_test -> Test features
        Y_test -> Test labels
        category_names -> label names (multi-output)
    """
    y_prediction = model.predict(X_test)
    class_report = classification_report(Y_test, y_prediction, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    """
   save_model
   This function saves trained model as Pickle file, to be loaded later.
   
   Arguments:
       model -> GridSearchCV object
       model_filepath -> destination path to save .pkl file
   
   """
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


def main():
    """ 
    main
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    """
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