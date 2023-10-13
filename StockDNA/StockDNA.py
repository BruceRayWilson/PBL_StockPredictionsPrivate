import os
import pandas as pd
import numpy as np

class StockDNA:
    # Class to process collected stock market data
    num_days = 10

    @staticmethod
    def exec(chunk_size):

        # Get list of all CSV files in the preprocessed_data directory
        files = [f for f in os.listdir('./preprocessed_data') if f.endswith('.csv')]

        # Create processed_data directory if it doesn't exist
        if not os.path.exists('./processed_data'):
            os.makedirs('./processed_data')

        # Process each CSV file
        for file in files:
            df = pd.read_csv(f'./preprocessed_data/{file}')

            # Get the number of days in the CSV file
            days_in_file = len(df.index)

            # Verify that the number of days is a multiple of chunk size
            if days_in_file % chunk_size != 0:
                raise ValueError(f"Number of days ({days_in_file}) is not a multiple of chunk size ({chunk_size})")

            # Bin Open,High,Low,Close,Tomorrow_Close into 5 bins and change the data
            for column in ['Open', 'High', 'Low', 'Close', 'Tomorrow_Close']:
                df[column] = pd.cut(df[column], bins=5, labels=['a', 'b', 'c', 'd', 'e'])

            # Bin Volume into 5 bins and change the data
            df['Volume'] = pd.cut(df['Volume'], bins=5, labels=['a', 'b', 'c', 'd', 'e'])

            # Save the processed data to the directory './processed_data'
            df.to_csv(f'./processed_data/{file}', index=False)

        print("Data processing completed.")

# # Get chunk size from StockPreprocessor
# chunk_size = StockPreprocessor.chunk_size

# # Process the data
# StockDna.process_data(chunk_size)