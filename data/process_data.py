import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    """
    Load and merge msessage and categories  dataset

    :param messages_filepath: str: Filepath for messages.csv file
    :param categories_filepath:: Filepath for categories.csv file
    :return:
    df: pandas Dataframe. containing the merged messages and categories dataset
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)

    # merge the two dataframes above
    df = pd.merge(messages, categories, how='left', on='id')

    return df

def clean_data(df):
    """
    Perform basic cleaning to the dataframe.:  splitting categories into seperate category columns , remove duplicates and coverting categories values
    into 0 and 1

    :param df: pandas Dataframe , containing the merged messages and categories dataset
    :return:  df: pandas Dataframe containing cleaned version of  input dataframe
    """
    # create a dataframe of the 36 individual category columns
    categories = df['categories'].str.split(';', expand=True)

    # select the first row of the categories dataframe
    row = categories.iloc[0,]
    # extract a list of column names for categories
    category_colnames = row.transform(lambda x: x[:-2]).tolist()
    # rename the columns of `categories`
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].transform(lambda x: x[-1])

        # convert column from string to numeric
        categories[column] = pd.to_numeric(categories[column])

    categories.drop(['child_alone'], axis=1, inplace=True)
    # drop the original categories column from `df`
    df.drop('categories', axis=1, inplace=True)
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1)
    # remove rows with related values of 2
    df = df[df.related != 2]
    # drop duplicates
    df.drop_duplicates(inplace=True)

    return df

def save_data(df, database_filename):
    """
    Save cleaned df to SQLite database.

    :param df: cleaned version of dataframe
    :param database_filename:  str. Filename for database
    :return:  None
    """
    engine = create_engine('sqlite:///'+database_filename)
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
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()