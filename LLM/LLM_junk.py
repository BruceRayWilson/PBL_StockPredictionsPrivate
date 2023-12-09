import os
import pandas as pd
import torch

class LLM:
    model = None
    tokenizer = None

    @classmethod
    def _initialize_model_and_tokenizer(cls):
        if cls.model is None or cls.tokenizer is None:
            # Initialize the model and tokenizer here
            pass

    @classmethod
    def predict_strings(cls, input_texts: list) -> list:
        '''Performs prediction on a batch of texts and returns the predicted class labels.'''
        cls._initialize_model_and_tokenizer()

        # Tokenize the input texts in batch
        tokens = cls.tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True)

        # Perform inference
        with torch.no_grad():
            outputs = cls.model(**tokens)

        # Post-process the outputs to get the predicted class labels
        predicted_classes = torch.argmax(outputs.logits, dim=1).tolist()

        return predicted_classes

    @staticmethod
    def predict(directory_path="test_data"):
        """
        Process all CSV files in the specified directory, add predictions as a new column,
        and save the updated DataFrame to a new file. Calculate and save the average 
        Tomorrow_Gain for PredictedClass 5, and delete old 'predicted_' files.
        """

        # Delete existing 'predicted_' files to avoid confusion
        for file in os.listdir(directory_path):
            if file.startswith("predicted_"):
                os.remove(os.path.join(directory_path, file))

        # List all CSV files in the directory that do not start with 'predicted_'
        csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv") and not file.startswith("predicted_")]

        # Initialize a list to hold the average gain results
        average_gains = []

        # Initialize model and tokenizer once, outside the loop
        LLM._initialize_model_and_tokenizer()

        # Iterate through each non-predicted CSV file
        for csv_file in csv_files:
            file_path = os.path.join(directory_path, csv_file)
            new_file_path = os.path.join(directory_path, f"predicted_{csv_file}")

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Ensure the 'Sentence' column exists in the DataFrame
            if 'Sentence' not in df.columns:
                continue

            # Predict in batches instead of single sentences
            predictions = LLM.predict_strings(df['Sentence'].tolist())

            # Add predictions as a new column to the DataFrame
            df['PredictedClass'] = predictions

            # Rename the 'Gain' column to 'Class' if it exists
            if 'Gain' in df.columns:
                df.rename(columns={'Gain': 'Class'}, inplace=True)

            # Calculate the average 'Tomorrow_Gain'
            avg_gain_cond = (df['Class'].isin([3, 4, 5])) & (df['PredictedClass'] == 5)
            average_gain = df.loc[avg_gain_cond, 'Tomorrow_Gain'].mean()
            average_gains.append((os.path.splitext(csv_file)[0], average_gain))

            # Save the updated DataFrame to a new CSV file
            df.to_csv(new_file_path, index=False)

        # Save the average gains to a CSV file
        gains_df = pd.DataFrame(average_gains, columns=['BaseFileName', 'AverageGain'])
        gains_df.to_csv(os.path.join(directory_path, "average_gains.csv"), index=False)
