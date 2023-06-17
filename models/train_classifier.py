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
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('Disaster_Data', engine)
    # There is 2 in the related column so we will replace it with 1
    df['related'] = df['related'].replace(2,1)
    # splitting into target and predicting variables
    X = df.message
    y = df[df.columns[4:]]
    category_names = y.columns
    return X, y, category_names


def tokenize(text):
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
    y_prediction = model.predict(X_test)
    class_report = classification_report(Y_test, y_prediction, target_names=category_names)
    print(class_report)


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as file:
        pickle.dump(model, file)


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