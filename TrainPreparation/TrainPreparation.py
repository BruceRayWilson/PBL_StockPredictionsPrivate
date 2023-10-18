import os
import pandas as pd
import numpy as np
from typing import List

# Awesome
class TrainPreparation:
    # Default directories
    DATA_DIR: str = 'processed_sentences'
    TARGET_DIR: str = 'train_data'
    # Categories for labeling data
    CATEGORY_LABELS: List[str] = ['really bad', 'bad', 'average', 'great', 'excellent']

    def __init__(self, data_dir: str = DATA_DIR, target_dir: str = TARGET_DIR) -> None:
        self.data_dir = data_dir
        self.target_dir = target_dir

    def exec(self) -> None:
        """Main execution method to prepare training data."""
        self.prepare_target_dir()
        all_gain_values = self.collect_gain_values()
        bin_values = self.calc_bin_edges(all_gain_values)
        self.process_files(bin_values)

    def prepare_target_dir(self) -> None:
        """Creates the target directory if it doesn't exist."""
        os.makedirs(self.target_dir, exist_ok=True)

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

    def process_files(self, bin_values: List[float]) -> None:
        """
        Process each file in the data directory:
        1. Categorize 'Tomorrow_Gain' using the bin values.
        2. Construct a new 'text' column.
        3. Save the modified dataframe to the target directory without the "_sentences" suffix.

        Args:
            bin_values (List[float]): Bin edges for categorizing the gain values.
        """
        files = os.listdir(self.data_dir)
        for file in files:
            df = pd.read_csv(os.path.join(self.data_dir, file))
            
            # Remove '_sentences' from filename if it exists
            new_filename = file.replace('_sentences', '')
            
            df['Gain'] = pd.cut(df['Tomorrow_Gain'], bins=bin_values, labels=self.CATEGORY_LABELS)
            df['text'] = '### Human: ' + df['Sentence'] + ' ### Assistant: ' + df['Gain'].astype(str)
            df.to_csv(os.path.join(self.target_dir, new_filename), index=False)

def main() -> None:
    train_prep = TrainPreparation()
    train_prep.exec()

if __name__ == "__main__":
    main()
