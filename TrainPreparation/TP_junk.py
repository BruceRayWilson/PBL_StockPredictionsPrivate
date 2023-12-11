import os
import pandas as pd

class FileProcessor:
    def __init__(self, data_dir, target_dir_train, target_dir_test, CATEGORY_LABELS):
        self.data_dir = data_dir
        self.target_dir_train = target_dir_train
        self.target_dir_test = target_dir_test
        self.CATEGORY_LABELS = CATEGORY_LABELS

    def process_files(self, args: Dict, bin_values: List[float]) -> None:
        """
        Process each file in the data directory:
        1. Categorize 'Tomorrow_Gain' using the bin values.
        2. Construct a new 'text' column.
        3. Save the modified dataframe to the target directory without the "_sentences" suffix.
        4. Append the last row of each file to a new DataFrame 'df_last_rows'.

        Args:
            args (Dict): Dictionary containing function arguments including 'predict_start_time'.
            bin_values (List[float]): Bin edges for categorizing the gain values.
        """
        files = os.listdir(self.data_dir)

        # Initialize a list for storing the last rows
        last_rows = []

        for file in files:
            df = pd.read_csv(os.path.join(self.data_dir, file))
            
            # Remove and save the last row
            last_row = df.tail(1)
            last_rows.append(last_row)

            # Remove the last row from the current DataFrame
            df = df.iloc[:-1]

            # Remove '_sentences' from filename if it exists
            new_filename = file.replace('_sentences', '')
            
            df['Gain'] = pd.cut(df['Tomorrow_Gain'], bins=bin_values, labels=self.CATEGORY_LABELS)
            selected_columns = ['Date', 'Sentence', 'Tomorrow_Gain', 'Gain']
            df = df[selected_columns]
            df['Sentence'] = df['Sentence'].astype(str)
            df['Gain'] = df['Gain'].astype(str)

            # Only take the train data.
            df_train = df[df['Date'] < args['predict_start_time']].copy()
            df_train.to_csv(os.path.join(self.target_dir_train, new_filename), index=False)

            # Only take the test data.
            df_test = df[df['Date'] >= args['predict_start_time']].copy()
            df_test.to_csv(os.path.join(self.target_dir_test, new_filename), index=False)

        # Create a DataFrame from the list of last rows
        df_last_rows = pd.DataFrame().append(last_rows, ignore_index=True)

        # Optionally, you can save df_last_rows to a file or return it
        # df_last_rows.to_csv('path_to_save/last_rows.csv', index=False)
        # return df_last_rows

# Example usage
# processor = FileProcessor(data_dir, target_dir_train, target_dir_test, CATEGORY_LABELS)
# processor.process_files(args, bin_values)
