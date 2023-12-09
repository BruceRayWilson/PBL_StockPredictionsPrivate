import pandas as pd
import os
from typing import List, Dict

class YourClassName:  # Replace with your actual class name
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

            # Assuming 'Timestamp' is the column to compare with predict_start_time
            df = df[pd.to_datetime(df['Timestamp']) < pd.to_datetime(args['predict_start_time'])]

            # Remove '_sentences' from filename if it exists
            new_filename = file.replace('_sentences', '')

            df['Gain'] = pd.cut(df['Tomorrow_Gain'], bins=bin_values, labels=self.CATEGORY_LABELS)
            selected_columns = ['Sentence', 'Tomorrow_Gain', 'Gain']
            df = df[selected_columns]
            df['Sentence'] = df['Sentence'].astype(str)
            df['Gain'] = df['Gain'].astype(str)

            df.to_csv(os.path.join(self.target_dir, new_filename), index=False)
