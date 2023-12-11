import os
import pandas as pd
import datetime
from typing import List, Union
from StockData.StockData import StockData

class PredictionResults:
    # Class to check stock market prediction results.

    directory: str = "."
    
    def __init__(self, directory=".", selected_file="", prediction_df=pd.DataFrame(), filtered_df=pd.DataFrame()):
        self.directory = directory
        self.selected_file = selected_file
        self.prediction_df = prediction_df
        self.filtered_df = filtered_df
        
    @staticmethod
    def filter_and_sort_df(df: pd.DataFrame) -> pd.DataFrame:
        # Check if necessary column exists
        if 'PredictedClass' not in df.columns or 'Symbol' not in df.columns:
            raise ValueError("Necessary columns missing in the dataframe.")
        df_filtered = df[df['PredictedClass'] == 5]
        return df_filtered.sort_values('Symbol')

    @staticmethod
    def calculate_gains(df: pd.DataFrame) -> pd.DataFrame:
        # Check if necessary columns exist
        if 'Close' not in df.columns or 'Open' not in df.columns:
            raise ValueError("Necessary columns missing in the dataframe.")
        df['PercentGain'] = 1 + ((df['Close'] - df['Open']) / df['Open'])
        df['DollarGain'] = df['Close'] - df['Open']
        return df
 
    @classmethod
    def get_file_selection(cls):
        ''' This method lists all csv files starting with 'predictions_output' and allows users to select one. '''
        files = [f for f in os.listdir(cls.directory) if f.startswith("predictions_output") and f.endswith(".csv")]
        # Check if files exist
        if not files:
            raise ValueError("No suitable files found in the directory.")
        print("Please select a file:")
        for idx, file in enumerate(files, 1):
            print(f"{idx}. {file}")
        selection = int(input("Enter selection: ")) - 1
        if selection < 0 or selection >= len(files):
            raise ValueError("Invalid selection.")
        return files[selection]

    @staticmethod
    def get_prediction_data(filename: str) -> pd.DataFrame:
        prediction_df = pd.read_csv(filename)
        return prediction_df

    @classmethod
    def init_instance(cls) -> None:
        ''' This method initializes the prediction data. '''

        # Get file selection from user and read the data into pandas dataframe
        cls.selected_file = cls.get_file_selection()
        cls.prediction_df = cls.get_prediction_data(cls.selected_file)
        
        # Filter rows where PredictedClass is 5 and sort the rows by 'Symbol'
        cls.filtered_df = cls.filter_and_sort_df(cls.prediction_df)
        cls.filtered_df['Symbol'].to_csv('predicted_symbols.csv', index=False)

    @classmethod
    def compute_gains(cls) -> None:
        ''' This method computes the gains. '''

        # Get start_time and end_time from 'Date' column
        start_time = pd.to_datetime(cls.filtered_df['Date']).min()
        end_time = start_time + datetime.timedelta(days=1)

        StockData.exec('predicted_symbols.csv', start_time, end_time)
        
        # Read the symbol data using stock_data object
        for symbol in cls.filtered_df['Symbol'].unique():
            # Get data for a particular stock symbol
            filename = os.path.join(cls.directory, f"data/{symbol}.csv")
            symbol_df = pd.read_csv(filename)
            
            # Match 'Date' from the stock_data with the prediction data
            symbol_df_filtered = symbol_df[symbol_df['Date'].isin(cls.filtered_df['Date'])]

            # Calculate 'PercentGain' and 'DollarGain'
            calculated_df = cls.calculate_gains(symbol_df_filtered)
            
            # Add 'PercentGain' and 'DollarGain' to the first data frame
            cls.filtered_df.loc[cls.filtered_df['Symbol'] == symbol, 'PercentGain'] = calculated_df['PercentGain'].values
            cls.filtered_df.loc[cls.filtered_df['Symbol'] == symbol, 'DollarGain'] = calculated_df['DollarGain'].values

        print("Calculation of gains done.")

    @classmethod
    def save_results(cls) -> None:
        ''' This method saves the results to a CSV file. '''
        cls.filtered_df.to_csv('final_predictions.csv', index=False)
        print("Results saved to final_predictions.csv.")

    @classmethod
    def exec(cls) -> None:
        ''' This classmethod checks the stock data prediction results and saves the final calculations in a CSV file. '''
        cls.init_instance()
        cls.compute_gains()
        cls.save_results()


if __name__ == '__main__':
    # Example usage:
    predictor = PredictionResults()
    predictor.exec()
