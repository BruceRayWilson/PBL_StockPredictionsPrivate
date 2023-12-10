import os
import pandas as pd
import numpy as np
from typing import List, Dict

class TrainPreparation:
    # Default directories
    DATA_DIR: str = 'processed_sentences'                       # <-- Input Directory
    TARGET_DIR_TRAIN: str = 'train_data'                        # <-- Train Output Directory
    TARGET_DIR_TEST:  str = 'test_data'                         # <-- Test Output Directory

    # Categories for labeling data
    CATEGORY_LABELS: List[str] = ['0', '1', '2', '3', '4', '5']

    def __init__(self, data_dir: str = DATA_DIR, target_dir_train: str = TARGET_DIR_TRAIN, target_dir_test: str = TARGET_DIR_TEST) -> None:
        self.data_dir = data_dir
        self.target_dir_train = target_dir_train
        self.target_dir_test = target_dir_test

    @classmethod
    def exec(cls, args) -> None:
        """Main execution method to prepare training data."""
        print("Training data preparation started...")
        instance = cls()  # Instantiate a new instance
        instance.prepare_dir(instance.target_dir_train)
        instance.prepare_dir(instance.target_dir_test)
        all_gain_values = instance.collect_gain_values()
        bin_values = instance.calc_bin_edges(all_gain_values)
        instance.process_files(args, bin_values)
        print("Training data preparation completed!")

    def prepare_dir(self, dir) -> None:
        """Creates the target directory if it doesn't exist and removes all files."""
        os.makedirs(dir, exist_ok=True)
        print(f"Output directory: {dir}")

        # Remove all files from the target directory
        for filename in os.listdir(dir):
            file_path = os.path.join(dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"Error deleting {file_path}: {str(e)}")


    def collect_gain_values(self) -> List[float]:
        """
        Collect 'Tomorrow_Gain' values from all files in the data directory.

        Returns:
            List[float]: A list containing all the 'Tomorrow_Gain' values from the files.
        """
        files = os.listdir(self.data_dir)
        all_gain_values = []
        for file in files:
            df = pd.read_csv(os.path.join(self.data_dir, file))
            all_gain_values += df['Tomorrow_Gain'].tolist()
        return all_gain_values

    def calc_bin_edges(self, all_gain_values: List[float]) -> List[float]:
        """
        Calculate bin edges based on the gain values for categorization.

        Args:
            all_gain_values (List[float]): List of all gain values.

        Returns:
            List[float]: Bin edges for categorizing the gain values.
        """
        sorted_gain = sorted(all_gain_values)
        bin_edges = [sorted_gain[i * (len(sorted_gain) - 1) // len(self.CATEGORY_LABELS)]
                    for i in range(len(self.CATEGORY_LABELS) + 1)]
        return bin_edges

    def process_files(self, args: Dict, bin_values: List[float]) -> None:
        """
        Process each file in the data directory:
        1. Categorize 'Tomorrow_Gain' using the bin values.
        2. Construct a new 'text' column.
        3. Save the modified dataframe to the target directory without the "_sentences" suffix.

        Args:
            args (Dict): Dictionary containing function arguments including 'predict_start_time'.
            bin_values (List[float]): Bin edges for categorizing the gain values.
        """
        files = os.listdir(self.data_dir)
        for file in files:
            df = pd.read_csv(os.path.join(self.data_dir, file))

            # Remove '_sentences' from filename if it exists
            new_filename = file.replace('_sentences', '')
            
            df['Gain'] = pd.cut(df['Tomorrow_Gain'], bins=bin_values, labels=self.CATEGORY_LABELS)
            selected_columns = ['Date', 'Sentence', 'Tomorrow_Gain', 'Gain']
            df = df[selected_columns]
            df['Sentence'] = df['Sentence'].astype(str)
            df['Gain'] = df['Gain'].astype(str)
            # print(df.dtypes)



            last_row = df.tail(1)
            print(last_row['Date'])
            print(f'args.predict_start_time: {args.predict_start_time}')



            # Only take the train data.
            df_train = df[df['Date'] < args.predict_start_time].copy()
            df_train.to_csv(os.path.join(self.target_dir_train, new_filename), index=False)

            # Only take the train data.
            df_test = df[df['Date'] >= args.predict_start_time].copy()
            df_test.to_csv(os.path.join(self.target_dir_test, new_filename), index=False)


def main() -> None:
    TrainPreparation.exec()

if __name__ == "__main__":
    main()
