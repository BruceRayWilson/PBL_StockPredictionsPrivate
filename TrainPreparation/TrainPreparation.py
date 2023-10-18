import os
import pandas as pd
import numpy as np
from typing import List

class TrainPreparation:
    def __init__(self, data_dir: str = 'processed_sentences', target_dir: str = 'train_data') -> None:
        self.data_dir = data_dir
        self.target_dir = target_dir
        self.category_labels = ['really bad', 'bad', 'average', 'great', 'excellent']

    def exec(self) -> None:
        self.prepare_target_dir()
        all_gain_values = self.collect_gain_values()
        bin_values = self.calc_bin_edges(all_gain_values)
        self.process_files(bin_values)

    def prepare_target_dir(self) -> None:
        os.makedirs(self.target_dir, exist_ok=True)
        
    def collect_gain_values(self) -> List[float]:
        files = os.listdir(self.data_dir)
        all_gain_values = []
        for file in files:
            df = pd.read_csv(os.path.join(self.data_dir, file))
            all_gain_values += df['Tomorrow_Gain'].tolist()
        return all_gain_values

    def calc_bin_edges(self, all_gain_values: List[float]) -> List[float]:
        sorted_gain = sorted(all_gain_values)
        bin_edges = [sorted_gain[i * len(sorted_gain) // len(self.category_labels)] 
                     for i in range(len(self.category_labels) + 1)]
        return bin_edges
    
    
    def process_files(self, bin_values: List[float]) -> None:
        files = os.listdir(self.data_dir)
        for file in files:
            df = pd.read_csv(os.path.join(self.data_dir, file))
            df['Gain'] = pd.cut(df['Tomorrow_Gain'], bins=bin_values, labels=self.category_labels)
            df['text'] = '### Human: ' + df['Sentence'] + ' ### Assistant: ' + df['Gain'].astype(str)
            df.to_csv(os.path.join(self.target_dir, file), index=False)

def main() -> None:
    train_prep = TrainPreparation()
    train_prep.exec()

if __name__ == "__main__":
    main()
