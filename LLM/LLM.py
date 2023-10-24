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







        data_path = "master_train_data/master.csv" #@param {type:"string"}
        text_column_name = "Sentence" #@param {type:"string"}
        label_column_name = "Gain" #@param {type:"string"}

        model_name = "distilbert-base-uncased" #@param {type:"string"}
        test_size = 0.2 #@param {type:"number"}
        num_labels = 5 #@param {type:"number"}

        df = pd.read_csv(data_path)
        df.head()

        from sklearn import preprocessing

        le = preprocessing.LabelEncoder()
        le.fit(df[label_column_name].tolist())
        df['label'] = le.transform(df[label_column_name].tolist())
        df.head()

        from sklearn.model_selection import train_test_split
        df_train, df_test = train_test_split(df, test_size=test_size)

        # Convert to Huggingface Dataset
        from datasets import Dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset  = Dataset.from_pandas(df_test)

        # Tokenizer
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            return tokenizer(examples[text_column_name], truncation=True)

        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        tokenized_test = test_dataset.map(preprocess_function, batched=True)

        # Initialize model
        from transformers import AutoModelForSequenceClassification

        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Train model
        from transformers import DataCollatorWithPadding
        from transformers import TrainingArguments, Trainer
        import evaluate
        import numpy as np

        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-4,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            num_train_epochs=5,
            weight_decay=0.01,
            evaluation_strategy = "epoch",
            logging_strategy="epoch"
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train,
            eval_dataset=tokenized_test,
            tokenizer=tokenizer,
            data_collator=data_collator,
            compute_metrics=compute_metrics

        )


        trainer.train()


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
