import os
import pandas as pd
import multiprocessing

class StockDNA:
    num_days = 7

    @staticmethod
    def process_single_csv(csv_file, chunk_size, config):
        stock_symbol = StockDNA.get_stock_symbol(csv_file)
        processed_df = StockDNA.process_csv_file(stock_symbol, chunk_size, config)
        sentences_df = StockDNA.create_sentences_from_data(processed_df)

        # Save the sentences dataframe to the 'processed_sentences' directory
        sentences_df.to_csv(f'./processed_sentences/{stock_symbol}_sentences.csv', index=False)

    @staticmethod
    def exec(chunk_size, config):
        print("Stock DNA...")

        # Get a list of all CSV files in the preprocessed_data directory
        csv_files = [file for file in os.listdir('./preprocessed_data') if file.endswith('.csv')]

        # Create the 'processed_data' directory if it doesn't exist
        if not os.path.exists('./processed_data'):
            os.makedirs('./processed_data')

        # Create the 'processed_sentences' directory if it doesn't exist
        if not os.path.exists('./processed_sentences'):
            os.makedirs('./processed_sentences')

        # Use multiprocessing to process CSV files concurrently
        with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
            # Pass an iterable of argument tuples to 'starmap'
            pool.starmap(StockDNA.process_single_csv, [(csv_file, chunk_size, config) for csv_file in csv_files])

        print("Stock DNA processing completed.")


    @staticmethod
    def get_stock_symbol(csv_file):
        # Extract the stock symbol from the CSV file's basename (filename without extension)
        return os.path.splitext(os.path.basename(csv_file))[0]


    @staticmethod
    def process_csv_file(stock_symbol, chunk_size, config):
        csv_file = stock_symbol + '.csv'

        # Read the CSV file
        df = pd.read_csv(f'./preprocessed_data/{csv_file}')

        # Get the number of days in the CSV file
        days_in_file = len(df)

        # Verify that the number of days is a multiple of chunk size
        if days_in_file % chunk_size != 0:
            raise ValueError(f"Number of days ({days_in_file}) is not a multiple of chunk size ({chunk_size})")
        
        # import string

        # Assuming config['rolling_windows'] is a list of numbers like [2, 5, 60, ...]
        rolling_windows = config['rolling_windows']

        # Base columns that do not depend on rolling windows
        base_columns = ['Open', 'High', 'Low', 'Close']

        # Dynamically create rolling close columns based on rolling_windows
        rolling_close_columns = [f'rolling_close_{window}' for window in rolling_windows]

        # Combine base columns with rolling close columns
        columns_to_bin = base_columns + rolling_close_columns
        # Now columns_to_bin contains columns like 'Open', 'High', 'Low', 'Close', 'rolling_close_2', 'rolling_close_5', 'rolling_close_60', ...

        label_list = [chr(97 + i) for i in range(26)]  # 'a' to 'z'
        label_list.extend([chr(97 + i) * 2 for i in range(26)])  # 'aa' to 'zz'

        # Shorten the label list.
        label_list = label_list[:32]
        bin_count = len(label_list)

        for column in columns_to_bin:
            df[column] = pd.cut(df[column], bins=bin_count, labels=label_list)

        # Bin 'Volume' into specified bins and change the data
        df['Volume'] = pd.cut(df['Volume'], bins=bin_count, labels=label_list)

        # Concatenate the binned columns into a new column named "word" with spaces and 'day'
        allColumns = columns_to_bin + ['Volume']
        df['Word'] = df[allColumns].apply(lambda x: ' '.join(map(str, x)) + ' day', axis=1)

        # Save the processed data to the 'processed_data' directory with the stock symbol as the filename
        processed_filename = f'./processed_data/{stock_symbol}.csv'
        df.to_csv(processed_filename, index=False)

        return df
    

    @staticmethod
    def create_sentences_from_data(df):
        # Initialize an empty list to store the sentences
        sentences = []
        
        # Loop through the DataFrame from the beginning to a point
        # where there are enough rows to create a sentence (num_days)
        for i in range(0, len(df) - StockDNA.num_days + 1):
            # Extract a window of data with 'num_days' rows
            window = df.iloc[i:i+StockDNA.num_days]
            
            # Concatenate the 'Word' values in the window to form a sentence
            sentence = " ".join(window['Word'].values)
            
            # Get the date from the last row of the window
            date = window['Date'].iloc[-1]
            
            # Get the 'Tomorrow_Gain' value from the last row of the window
            tomorrow_gain = window['Tomorrow_Gain'].iloc[-1]
            
            # Append the date, tomorrow's gain, and the sentence to the 'sentences' list
            sentences.append([date, tomorrow_gain, sentence])
        
        # Create a DataFrame from the list of sentences with specified column names
        return pd.DataFrame(sentences, columns=['Date', 'Tomorrow_Gain', 'Sentence'])

if __name__ == "__main__":
    chunk_size = 10
    StockDNA.exec(chunk_size)
