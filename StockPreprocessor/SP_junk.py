import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler

class DataPreprocessor:
    def __init__(self, input_directory, output_directory, rolling_windows, chunk_size):
        self.input_directory = input_directory
        self.output_directory = output_directory
        self.rolling_windows = rolling_windows
        self.chunk_size = chunk_size

    def preprocess_file(self, file):
        # Read CSV file
        df = pd.read_csv(os.path.join(self.input_directory, file))

        # Keep only the necessary fields
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']]

        # Calculate Tomorrow_Gain as tomorrow's close divided by tomorrow's open
        df['Tomorrow_Gain'] = df['Close'].shift(-1) / df['Open'].shift(-1)

        # Calculate rolling mean for each window size
        for window in self.rolling_windows:
            df[f'rolling_close_{window}'] = df['Close'].rolling(window=window).mean()

        # Drop rows with NaN values
        df.dropna(inplace=True)

        # Divide data into chunks of the specified size in reverse order
        chunks = [df[i:i + self.chunk_size] for i in range(0, df.shape[0], self.chunk_size)]

        # Dynamically generate rolling close column names based on rolling windows
        rolling_close_columns = [f'rolling_close_{window}' for window in self.rolling_windows]

        # Define the static columns
        static_columns = ['Date', 'Open', 'High', 'Low', 'Close', 'Tomorrow_Gain', 'Volume']

        # Initialize an empty DataFrame to store the concatenated chunks
        result_df = pd.DataFrame()

        for chunk in chunks:
            # Extract only the necessary columns
            ohlc_data = chunk[['Open', 'High', 'Low', 'Close'] + rolling_close_columns]
            volume_data = chunk[['Volume']]
            gain_data = chunk[['Tomorrow_Gain']]

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

# Usage example:
# preprocessor = DataPreprocessor(input_directory='path_to_input_dir',
#                                 output_directory='path_to_output_dir',
#                                 rolling_windows=[5, 10, 20],
#                                 chunk_size=100)
# preprocessor.preprocess_file('example.csv')
