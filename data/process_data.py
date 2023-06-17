import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    load_data
    Load Messages Data with Categories Function
    
    Arguments:
        messages_filepath -> Path to the CSV file containing messages
        categories_filepath -> Path to the CSV file containing categories
    Output:
        df -> Combined data containing messages and categories
    """
    
    # load the datasets
    categories = pd.read_csv(categories_filepath)
    messages = pd.read_csv(messages_filepath)
    # merge both the datasets
    df = pd.merge(messages, categories, on ='id')
    
    return df


def clean_data(df):
    """
    clean_data
    
    Arguments:
        df -> Combined data containing messages and categories
    Outputs:
        df -> Combined data containing messages and categories with categories cleaned up
    """
    
    # create a dataframe of the 36 individual category columns
    categories = df["categories"].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.head(1)

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.apply(lambda lis: lis.str[:-2]).values.tolist()
    # converting nested list to flat list
    category_colnames = list(np.concatenate(category_colnames))
    # rename the columns of `categories`
    categories.columns = category_colnames
        # Convert category values to just numbers 0 or 1.
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace = True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    
    # There is 2 in the related column so we will drop all the 2s as there are only 193 values
    df = df[df.related != 2]
    
    # drop duplicates
    df = df.drop_duplicates()
    
    return df


def save_data(df, database_filename):
    """
   save_data
   saving data to SQLite Database
   
   Arguments:
       df -> Combined data containing messages and categories with categories cleaned up
       database_filename -> Path to SQLite destination database
   """
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Disaster_Data', engine, index=False, if_exists='replace')  


def main():
    """
   Main
       1) Load Messages Data with Categories
       2) Clean Categories Data
       3) Save Data to SQLite Database
   """
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()