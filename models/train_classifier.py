import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.externals import joblib

def load_data(database_filepath):
    '''
    Load data from database file path, output feature set, target and target categories
    
    :param database_filepath: database file path
    '''
    # load data from database
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql_table('disaster_response',engine)
    X = df['message']
    Y = df.drop(columns=['id','message','original','genre'])
    
    return X, Y, Y.columns


def tokenize(text):
    '''
    Tokenize, lemmatize, normalize, strip, remove stop words from the text
    
    :param text: input text
    '''
    # Initialization
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stopWords = set(stopwords.words('english'))
    
    # Get clean tokens after lemmatization, normalization, stripping and stop words removal
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        if tok not in stopWords:
            clean_tokens.append(clean_tok)

    return clean_tokens


class TextLengthExtractor(BaseEstimator, TransformerMixin):
    '''
    An estimator that can count the text length of each cell in the X
    
    '''
    def fit(self, X, y=None):
    	'''
    	Return self
    	'''
    	return self

    def transform(self, X):
    	'''
    	Count the text length of each cell in the X
    	'''
    	X_length = pd.Series(X).str.len()
    	return pd.DataFrame(X_length)


def build_model():
    '''
    Build a pipeline with TFIDF DTM, length of text column, and a random forest classifier. Grid search on the `use_idf` from tf_idf and `n_estimators` from random forest classifier to find the best model
    
    :param text: input text
    '''
    # Build out the pipeline
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('text_len', TextLengthExtractor())
        ])),

        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs=-1)))
    ])
    # Set up the search grid
    parameters = {
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 100]
    }
    # Initialize GridSearch cross validation object
    cv = GridSearchCV(pipeline, param_grid=parameters,n_jobs=-1)

    return cv

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Evaluate the model performance of each category target column
    
    :param model: model object
    :param X_test: test feature set
    :param Y_test: test target set
    :param category_names: target category names
    '''
    # Use model to predict
    Y_pred = model.predict(X_test)
    # Turn prediction into DataFrame
    Y_pred = pd.DataFrame(Y_pred,columns=category_names)
    # For each category column, print performance
    for col in category_names:
        print(f'Column Name:{col}\n')
        print(classification_report(Y_test[col],Y_pred[col]))


def save_model(model, model_filepath):
    '''
    Save model to a pickle file
    
    :param model: model object
    :param model_filepath: model output file path
    '''
    joblib.dump(model, model_filepath) 


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