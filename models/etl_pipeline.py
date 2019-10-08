import sys
import numpy as np
import pandas as pd
import sqlalchemy as db

def load_data(messages_file, categories_file):
     # Load messages dataset
    data_folder = '../data/'
    messages = pd.read_csv(messages_file)
    categories = pd.read_csv(categories_file)

    # Merge datasets
    df = messages.merge(categories, how = 'left', on = ['id'])
    
    return df



def clean_data(df):
     # Create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand = True)
    
    # Select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    
    # Rename the columns of `categories`
    categories.columns = category_colnames
    
    # Convert  category values to numeric values
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1:])
        
        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])
    
    # Drop the original categories column from `df`
    df.drop('categories', axis = 1, inplace = True)
    
    
    # Concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis = 1)
    
    # Drop duplicates
    df.drop_duplicates()
    
    return df



def save_data(df, database_filename):
    '''
        Function for saving data with given filepath
    '''
    engine = db.create_engine('sqlite:///' + database_filename)
    df.to_sql('Messages', engine, index=False, if_exists='replace')  


def main():
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
        print('Please provide the filepaths of the messages,  categories '\
              'Database file name respectively')


if __name__ == '__main__':
    main()