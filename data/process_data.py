import sys
import pandas as pd
from sqlalchemy import create_engine

class DisasterResponsePipeline:
    def __init__(self, message_filepath: str, category_filepath: str, database_filename: str):
        """
        Initialize the pipeline with file paths for input data and SQLite database.
        
        :param message_filepath: Path to the messages CSV file
        :param category_filepath: Path to the categories CSV file
        :param database_filename: SQLite database filename
        """
        self.message_filepath = message_filepath
        self.category_filepath = category_filepath
        self.database_filename = database_filename
        self.data = None

    def import_data(self):
        """
        Load message and category datasets from CSV files, then merge them into a single DataFrame.
        """
        message_data = pd.read_csv(self.message_filepath)
        category_data = pd.read_csv(self.category_filepath)

        self.data = message_data.merge(category_data, on=["id"], how="left")

    def preprocess_data(self):
        """
        Perform cleaning operations on the data:
        - Split categories into separate columns
        - Convert category values to binary (0 or 1)
        - Remove duplicates
        """
        extracted_categories = self.data["categories"].str.split(";", expand=True)
        first_row = extracted_categories.iloc[0]
        new_column_names = first_row.apply(lambda x: x[:-2])
        extracted_categories.columns = new_column_names

        for column in extracted_categories:
            extracted_categories[column] = extracted_categories[column].str[-1].astype(int)
        
        self.data.drop("categories", axis=1, inplace=True)
        self.data = self.data.merge(extracted_categories, how="left", left_index=True, right_index=True)
        
        self.data.drop_duplicates(inplace=True)
        self.data = self.data[self.data["related"].between(0, 1)]

    def save_to_db(self):
        """
        Store the cleaned and processed data in an SQLite database for further use.
        """
        engine = create_engine("sqlite:///" + self.database_filename)
        self.data.to_sql('DisasterResponse', engine, index=False, if_exists='replace')  
    
    def run_pipeline(self):
        """
        Execute the sequence of ETL steps: Load data, preprocess and clean data, save to SQLite database.
        """
        print('Step 1: Importing datasets...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(self.message_filepath, self.category_filepath))
        self.import_data()

        print('Step 2: Cleaning and preprocessing the data...')
        self.preprocess_data()
        
        print('Step 3: Storing the cleaned data into the database...\n    DATABASE: {}'.format(self.database_filename))
        self.save_to_db()
        
        print('Pipeline completed successfully. Cleaned data has been saved to the database.')

if __name__ == '__main__':
    if len(sys.argv) == 4:
        message_filepath, category_filepath, database_filepath = sys.argv[1:]
        etl_pipeline = DisasterResponsePipeline(message_filepath, category_filepath, database_filepath)
        etl_pipeline.run_pipeline()
    else:
        print('Input Error: Please provide the filepaths for the messages and categories '\
              'datasets as the first two arguments, and the filepath for the SQLite database '\
              'to save the cleaned data as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv DisasterResponse.db')
