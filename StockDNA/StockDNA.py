import os
import pandas as pd

class StockDNA:
    num_days = 10

    @staticmethod
    def exec(chunk_size):
        # Get a list of all CSV files in the preprocessed_data directory
        csv_files = [file for file in os.listdir('./preprocessed_data') if file.endswith('.csv')]

        # Create the 'processed_data' directory if it doesn't exist
        if not os.path.exists('./processed_data'):
            os.makedirs('./processed_data')

        # Process each CSV file
        for csv_file in csv_files:
            StockDNA.process_csv_file(csv_file, chunk_size)

        print("Data processing completed.")

    @staticmethod
    def process_csv_file(csv_file, chunk_size):
        # Read the CSV file
        df = pd.read_csv(f'./preprocessed_data/{csv_file}')

        # Get the number of days in the CSV file
        days_in_file = len(df)

        # Verify that the number of days is a multiple of chunk size
        if days_in_file % chunk_size != 0:
            raise ValueError(f"Number of days ({days_in_file}) is not a multiple of chunk size ({chunk_size})")

        # Bin 'Open', 'High', 'Low', 'Close', 'Tomorrow_Close' into 5 bins and change the data
        columns_to_bin = ['Open', 'High', 'Low', 'Close', 'Tomorrow_Close']
        for column in columns_to_bin:
            df[column] = pd.cut(df[column], bins=5, labels=['a', 'b', 'c', 'd', 'e'])

        # Bin 'Volume' into 5 bins and change the data
        df['Volume'] = pd.cut(df['Volume'], bins=5, labels=['a', 'b', 'c', 'd', 'e'])

        # Save the processed data to the 'processed_data' directory
        df.to_csv(f'./processed_data/{csv_file}', index=False)

# if __name__ == "__main__":
#     chunk_size = 10
#     StockDNA.process_csv_files(chunk_size)


# # Get chunk size from StockPreprocessor
# chunk_size = StockPreprocessor.chunk_size

# # Process the data
# StockDna.process_data(chunk_size)