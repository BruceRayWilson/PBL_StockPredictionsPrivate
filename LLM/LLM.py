from datasets import load_dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd

class LLM:
    """Class to manage Linear Level Models (LLM) for stock market data analysis"""

    @staticmethod
    def train() -> None:
        '''Trains LLM, evaluates on a validation set, and prints evaluation metrics.'''
        print("LLM training and evaluation...")

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias='none',
            task_type='CASUAL_LM',
        )

        # help(load_dataset)

        # Load the dataset for training and test
        dataset = load_dataset("master_train_data", split=None)
        print(f"dataset: {dataset}")
        print(f"\n\n\t\tTraining dataset size: {len(dataset['train'])}\n\n")
        # print(f"\n\n\t\tTest dataset size: {len(dataset['test'])}\n\n")

        # Initialize the trainer
        trainer = SFTTrainer(
            "facebook/opt-350m",
            train_dataset=dataset['train'],
            dataset_text_field="text",
            max_seq_length=128,
            peft_config=peft_config
        )

        # Train the model
        trainer.train()

        # # Load the pre-trained model and tokenizer for evaluation
        # model_name = "facebook/opt-350m"  # Replace with the model you trained
        # model = AutoModelForSequenceClassification.from_pretrained(model_name)
        # tokenizer = AutoTokenizer.from_pretrained(model_name)

        # # Evaluate the model on the validation dataset
        # valid_inputs = [example["text"] for example in dataset['test']]
        # valid_labels = [example["label"] for example in dataset['test']]

        # # Tokenize the validation inputs
        # valid_tokens = tokenizer(valid_inputs, return_tensors="pt", padding=True, truncation=True)

        # # Perform inference on the validation set
        # with torch.no_grad():
        #     valid_output = model(**valid_tokens)

        # # Post-process the output to get predicted class labels
        # valid_predictions = torch.argmax(valid_output.logits, dim=1).tolist()

        # # Calculate accuracy and other evaluation metrics
        # accuracy = accuracy_score(valid_labels, valid_predictions)
        # classification_rep = classification_report(valid_labels, valid_predictions)

        # # Print evaluation results
        # print(f"Validation Accuracy: {accuracy}")
        # print("Classification Report:\n", classification_rep)

        # # Plot the confusion matrix
        # LLM.plot_confusion_matrix(valid_labels, valid_predictions)

    @staticmethod
    def plot_confusion_matrix(true_labels, predicted_labels, class_names=None):
        '''Plots a confusion matrix.'''
        cm = confusion_matrix(true_labels, predicted_labels)
        if class_names is None:
            class_names = list(map(str, np.unique(true_labels)))

        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix')
        plt.colorbar()
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.show()

    @staticmethod
    def predict_string(input_text: str) -> int:
        '''Performs prediction using the trained LLM model and returns the predicted class label.'''
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        import torch

        # Load the pre-trained model and tokenizer
        model_name = "Test_002"
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Tokenize the input text
        tokens = tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True)

        # Perform inference
        with torch.no_grad():
            output = model(**tokens)

        # Post-process the output to get the predicted class label
        predicted_class = torch.argmax(output.logits, dim=1).item()

        return predicted_class

    @staticmethod
    def predict(directory_path="test_data"):
        """
        Process all CSV files in the specified directory using the 'Sentence' column
        and call the predict method for each sentence.

        Args:
            directory_path (str): The path to the directory containing CSV files.
        """
        # List all CSV files in the specified directory
        csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv")]

        # Iterate through each CSV file
        for csv_file in csv_files:
            file_path = os.path.join(directory_path, csv_file)

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Ensure the 'Sentence' column exists in the DataFrame
            if 'Sentence' not in df.columns:
                print(f"Warning: 'Sentence' column not found in {csv_file}. Skipping...")
                continue

            # Process each sentence in the 'Sentence' column
            for sentence in df['Sentence']:
                predicted_class = LLM.predict_string(sentence)
                # Do something with the predicted_class (e.g., store it or print it)
                print(f"Sentence: {sentence}")
                print(f"Predicted Class: {predicted_class}\n")
