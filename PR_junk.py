import os
import pandas as pd
import datetime

class PredictionResults:
    """
    Class to process and analyze stock market prediction results.

    Attributes:
        directory (str): Directory where the data files are located.
        selected_file (str): File selected for analysis.
        prediction_df (pd.DataFrame): DataFrame containing prediction data.
        filtered_df (pd.DataFrame): DataFrame filtered based on prediction criteria.
    """

    def __init__(self, directory="."):
        """
        Initializes the PredictionResults object.

        Args:
            directory (str): Directory where the data files are located.
        """
        self.directory = directory
        self.selected_file = ""
        self.prediction_df = pd.DataFrame()
        self.filtered_df = pd.DataFrame()

    def get_file_selection(self) -> str:
        """
        Lists all suitable csv files and prompts user to select one.

        Returns:
            str: Filename selected by the user.
        """
        files = [f for f in os.listdir(self.directory) if f.startswith("predictions_output") and f.endswith(".csv")]
        if not files:
            raise ValueError("No suitable files found in the directory.")

        print("Please select a file:")
        for idx, file in enumerate(files, 1):
            print(f"{idx}. {file}")

        selection = int(input("Enter selection: ")) - 1
        if selection < 0 or selection >= len(files):
            raise ValueError("Invalid selection.")

        return files[selection]

    def get_prediction_data(self, filename: str) -> pd.DataFrame:
        """
        Reads prediction data from the specified file.

        Args:
            filename (str): Filename to read data from.

        Returns:
            pd.DataFrame: DataFrame containing the prediction data.
        """
        return pd.read_csv(os.path.join(self.directory, filename))

    def filter_and_sort_df(self) -> None:
        """
        Filters and sorts the prediction DataFrame based on specific criteria.
        """
        if 'PredictedClass' not in self.prediction_df.columns or 'Symbol' not in self.prediction_df.columns:
            raise ValueError("Necessary columns missing in the dataframe.")

        self.filtered_df = self.prediction_df[self.prediction_df['PredictedClass'] == 5].sort_values('Symbol')

    def calculate_gains(self, symbol: str) -> None:
        """
        Calculates the gains for a given stock symbol.

        Args:
            symbol (str): The stock symbol to calculate gains for.
        """
        filename = os.path.join(self.directory, f"data/{symbol}.csv")
        symbol_df = pd.read_csv(filename)

        if 'Close' not in symbol_df.columns or 'Open' not in symbol_df.columns:
            raise ValueError("Necessary columns missing in the stock data.")

        symbol_df['PercentGain'] = 1 + ((symbol_df['Close'] - symbol_df['Open']) / symbol_df['Open'])
        symbol_df['DollarGain'] = symbol_df['Close'] - symbol_df['Open']

        # Merge calculated gains back into filtered_df
        self.filtered_df = self.filtered_df.merge(symbol_df[['Date', 'PercentGain', 'DollarGain']], on='Date', how='left')

    def compute_gains(self) -> None:
        """
        Computes the gains for all filtered stock predictions.
        """
        symbols = self.filtered_df['Symbol'].unique()
        for symbol in symbols:
            self.calculate_gains(symbol)

    def save_results(self) -> None:
        """
        Saves the final calculated results to a CSV file.
        """
        self.filtered_df.to_csv(os.path.join(self.directory, 'final_predictions.csv'), index=False)
        print("Results saved to final_predictions.csv.")

    def exec(self) -> None:
        """
        Main method to execute the process of reading, analyzing, and saving stock prediction results.
        """
        self.selected_file = self.get_file_selection()
        self.prediction_df = self.get_prediction_data(self.selected_file)
        self.filter_and_sort_df()
        self.compute_gains()
        self.save_results()


if __name__ == '__main__':
    # Example usage:
    directory = "."  # specify the directory if different
    predictor = PredictionResults(directory)
    predictor.exec()
