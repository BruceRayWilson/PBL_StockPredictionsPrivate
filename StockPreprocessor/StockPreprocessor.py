import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from config_loader import load_config



class StockPreprocessor:
    chunk_size = 42

    def __init__(self):
        self.input_directory = 'data'
        self.output_directory = 'preprocessed_data'
        self.config = load_config()
        self.rolling_windows = self.config['rolling_windows']

    def create_output_directory(self):
        if not os.path.exists(self.output_directory):
            os.makedirs(self.output_directory)

    def load_csv_files(self):
        files = [f for f in os.listdir(self.input_directory) if f.endswith('.csv')]
        return files

    def preprocess_file(self, file):
        # Read CSV file
        df = pd.read_csv(os.path.join(self.input_directory, file))

        # Keep only the necessary fields
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Calculate Tomorrow_Gain as tomorrow's close divided by tomorrow's open
        df['Tomorrow_Gain'] = df['Close'].shift(-1) / df['Open'].shift(-1)

        # List of rolling windows
        rolling_windows = self.rolling_windows

        # Calculate rolling mean for each window size
        for window in rolling_windows:
            df[f'rolling_close_{window}'] = df['Close'].rolling(window=window).mean()

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # Divide data into chunks of the specified size in reverse order
        chunks = [df[i:i + self.chunk_size] for i in range(df.shape[0] - self.chunk_size, -1, -self.chunk_size)]

        # Dynamically generate rolling close column names based on rolling windows
        rolling_close_columns = [f'rolling_close_{window}' for window in self.rolling_windows]

        # Define the static columns
        static_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Tomorrow_Gain', 'Volume']

        # Initialize an empty DataFrame to store the concatenated chunks
        # with dynamically generated rolling close columns
        result_df = pd.DataFrame(columns=static_columns + rolling_close_columns)

        for chunk in chunks:
            # Extract only the necessary columns, including dynamically determined rolling close columns
            ohlc_data = chunk[['Open', 'High', 'Low', 'Close'] + rolling_close_columns]
            volume_data = chunk[['Volume']]

            # Create scalers for normalization
            ohlc_scaler = MinMaxScaler()
            volume_scaler = MinMaxScaler()

            # Normalize the OHLC and volume data
            chunk.loc[:, ['Open', 'High', 'Low', 'Close'] + rolling_close_columns] = ohlc_scaler.fit_transform(ohlc_data)
            chunk.loc[:, ['Volume']] = volume_scaler.fit_transform(volume_data)

            # Append the chunk to the result DataFrame
            result_df = pd.concat([result_df, chunk])

        # Sort the result DataFrame by 'Date'
        result_df.sort_values(by=['Date'], inplace=True)

        # Save the preprocessed data to a new CSV file
        result_df.to_csv(os.path.join(self.output_directory, file), index=False)


    def preprocess_data(self):
        print("Preprocessing data...")
        self.create_output_directory()
        files = self.load_csv_files()

        for file in files:
            self.preprocess_file(file)

        print("Preprocessing completed.")

    @classmethod
    def exec(cls):
        stock_preprocessor = cls()
        stock_preprocessor.preprocess_data()

if __name__ == "__main__":
    StockPreprocessor.exec()
