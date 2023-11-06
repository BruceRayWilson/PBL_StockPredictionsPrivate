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
import shutil
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from peft import LoraConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, TrainingArguments, Trainer
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import evaluate

class LLM:
    """Class to manage Linear Level Models (LLM) for stock market data analysis"""

    model = None
    tokenizer = None

    @classmethod
    def _initialize_model_and_tokenizer(cls):
        if cls.model is None or cls.tokenizer is None:
            model_name = "model_001"
            cls.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            cls.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @staticmethod
    def train() -> None:
        '''Trains LLM, evaluates on a validation set, and prints evaluation metrics.'''
        print("\nLLM training and evaluation...")

        # Verify that 'results' is a directory and delete it if it exists
        if os.path.isdir("results"):
            shutil.rmtree("results")

        # peft_config = LoraConfig(
        #     r=16,
        #     lora_alpha=32,
        #     lora_dropout=0.05,
        #     bias='none',
        #     task_type='CASUAL_LM',
        # )

        data_path = "master_train_data/master.csv" #@param {type:"string"}
        text_column_name = "Sentence" #@param {type:"string"}
        label_column_name = "Gain" #@param {type:"string"}

        model_name = "distilbert-base-uncased" #@param {type:"string"}
        test_size = 0.2 #@param {type:"number"}
        num_labels = 6 #@param {type:"number"}

        df = pd.read_csv(data_path)
        # print(f"df.head(): {df.head()}")

        le = preprocessing.LabelEncoder()
        le.fit(df[label_column_name].tolist())
        df['label'] = le.transform(df[label_column_name].tolist())
        # print(f"df.head(): {df.head()}")

        df_train, df_test = train_test_split(df, test_size=test_size)

        # Convert to Huggingface Dataset
        train_dataset = Dataset.from_pandas(df_train)
        test_dataset  = Dataset.from_pandas(df_test)

        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        def preprocess_function(examples):
            return tokenizer(examples[text_column_name], truncation=True)

        tokenized_train = train_dataset.map(preprocess_function, batched=True)
        tokenized_test = test_dataset.map(preprocess_function, batched=True)

        # Initialize model
        model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

        # Train model
        data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

        metric = evaluate.load("accuracy")

        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return metric.compute(predictions=predictions, references=labels)

        training_args = TrainingArguments(
            output_dir="./results",
            learning_rate=2e-4,
            per_device_train_batch_size=256,
            per_device_eval_batch_size=256,
            num_train_epochs=1,
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
        trainer.save_model('model_001')

        # Compute and print classification reports
        preds = trainer.predict(tokenized_train)
        preds = np.argmax(preds.predictions, axis=1)
        GT = df_train['label'].tolist()
        print("Classification Report (Train):")
        print(classification_report(GT, preds))

        preds = trainer.predict(tokenized_test)
        preds = np.argmax(preds.predictions, axis=1)
        GT = df_test['label'].tolist()
        print("Classification Report (Test):")
        print(classification_report(GT, preds))

        # Call the plot_confusion_matrix method
        LLM.plot_confusion_matrix(GT, preds, class_names=le.classes_)
        LLM.save_labels_to_csv(GT, preds)
        LLM.plot_bar_graph(GT, preds)


    @staticmethod
    def plot_bar_graph(true_labels, predicted_labels, save_path="bar_graph.png"):
        '''Plots a bar graph of true vs. predicted label counts and saves the plot to a file.'''

        # Determine unique labels and their counts
        unique_labels = np.unique(np.concatenate((true_labels, predicted_labels)))
        true_counts = [np.sum(true_labels == label) for label in unique_labels]
        predicted_counts = [np.sum(predicted_labels == label) for label in unique_labels]

        # Bar width
        bar_width = 0.35
        r1 = np.arange(len(unique_labels))
        r2 = [x + bar_width for x in r1]

        # Create the bar graph
        plt.figure(figsize=(10, 6))
        plt.bar(r1, true_counts, color='b', width=bar_width, edgecolor='grey', label='True Labels')
        plt.bar(r2, predicted_counts, color='r', width=bar_width, edgecolor='grey', label='Predicted Labels')

        # Title & subtitle
        plt.title('True vs. Predicted Label Counts', fontweight='bold')
        plt.xlabel('Labels', fontweight='bold')
        plt.ylabel('Counts', fontweight='bold')
        plt.xticks([r + bar_width for r in range(len(unique_labels))], unique_labels)

        # Create legend & show the graph
        plt.legend()

        # Save the figure
        plt.tight_layout()
        plt.savefig(save_path)

        # Close the figure
        plt.close()

    @staticmethod
    def plot_confusion_matrix(true_labels, predicted_labels, class_names=None, save_path="confusion_matrix.png"):
        '''Plots a confusion matrix.'''
        cm = confusion_matrix(true_labels, predicted_labels)
        if class_names is None:
            class_names = list(map(str, np.unique(true_labels)))

        # Convert counts to percentages
        cm_percentage = cm.astype('float') / cm.sum(axis=0)[np.newaxis, :] * 100
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm_percentage, interpolation='nearest', cmap=plt.cm.Blues)
        plt.title('Confusion Matrix (in %)')
        plt.colorbar(format='%1.2f%%')
        tick_marks = np.arange(len(class_names))
        plt.xticks(tick_marks, class_names, rotation=45)
        plt.yticks(tick_marks, class_names)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')

        # Display the percentage for each square in the matrix
        thresh = cm_percentage.max() / 2.  # threshold to determine the color of the text (for visibility)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, "{:.2f}%".format(cm_percentage[i, j]),
                         horizontalalignment="center",
                         color="white" if cm_percentage[i, j] > thresh else "black")

        # Save the figure
        plt.tight_layout()  # This will ensure that labels and titles don't get cut off
        plt.savefig(save_path)

        plt.show()

        # Close the figure
        plt.close()

        
    @staticmethod
    def save_labels_to_csv(true_labels, predicted_labels, file_path="labels.csv"):
        '''Saves the true and predicted labels to a CSV file.'''

        # Create a DataFrame from the labels
        df = pd.DataFrame({
            'True_Labels': true_labels,
            'Predicted_Labels': predicted_labels
        })

        # Save the DataFrame to a CSV file
        df.to_csv(file_path, index=False)


    @classmethod
    def predict_string(cls, input_text: str) -> int:
        '''Performs prediction using the trained LLM model and returns the predicted class label.'''
        cls._initialize_model_and_tokenizer()

        # Tokenize the input text
        tokens = cls.tokenizer.encode_plus(input_text, return_tensors="pt", padding=True, truncation=True)

        # Perform inference
        with torch.no_grad():
            output = cls.model(**tokens)

        # Post-process the output to get the predicted class label
        predicted_class = torch.argmax(output.logits, dim=1).item()

        return predicted_class

    import os
    import pandas as pd

    @staticmethod
    def predict(directory_path="test_data"):
        """
        Process all CSV files in the specified directory using the 'Sentence' column,
        add predictions as a new column, and save the updated DataFrame to a new file.
        Also calculate and save the average Tomorrow_Gain for PredictedClass 5, delete
        old 'predicted_' files, and ensure they are not processed.
        """

        # Delete existing 'predicted_' files to avoid confusion
        for file in os.listdir(directory_path):
            if file.startswith("predicted_"):
                os.remove(os.path.join(directory_path, file))
                print(f"Deleted {file}")

        # List all CSV files in the specified directory that do not start with 'predicted_'
        csv_files = [file for file in os.listdir(directory_path) if file.endswith(".csv") and not file.startswith("predicted_")]

        # Initialize a list to hold the average gain results
        average_gains = []

        # Iterate through each non-predicted CSV file
        for csv_file in csv_files:
            print(f"\nProcessing {csv_file}...")
            file_path = os.path.join(directory_path, csv_file)
            new_file_path = os.path.join(directory_path, f"predicted_{csv_file}")

            # Read the CSV file into a DataFrame
            df = pd.read_csv(file_path)

            # Ensure the 'Sentence' column exists in the DataFrame
            if 'Sentence' not in df.columns:
                print(f"Warning: 'Sentence' column not found in {csv_file}. Skipping...")
                continue

            # Make predictions for each sentence in the 'Sentence' column
            predictions = [LLM.predict_string(sentence) for sentence in df['Sentence']]

            # Add predictions as a new column to the DataFrame
            df['PredictedClass'] = predictions

            # Rename the 'Gain' column to 'Class'.
            df.rename(columns={'Gain': 'Class'}, inplace=True)

            # Calculate the average 'Tomorrow_Gain' for rows where 'PredictedClass' is 5
            # and 'Class' is in [3, 4, 5]
            class_condition = df['Class'].isin([3, 4, 5])
            predicted_class_condition = df['PredictedClass'] == 5
            average_gain = df.loc[class_condition & predicted_class_condition, 'Tomorrow_Gain'].mean()

            # Append the average gain and the base file name to the list
            base_file_name = os.path.splitext(csv_file)[0]
            average_gains.append((base_file_name, average_gain))

            # Save the updated DataFrame to a new CSV file
            df.to_csv(new_file_path, index=False)
            print(f"Updated file saved to {new_file_path}")

        # Create a DataFrame from the average gains list
        gains_df = pd.DataFrame(average_gains, columns=['BaseFileName', 'AverageGain'])

        # Calculate the average of the average gains
        overall_avg_gain = gains_df['AverageGain'].mean()

        print(f'Overall Average: {overall_avg_gain}')

        # Append a new row to gains_df with the average of averages
        # gains_df = gains_df.append({'BaseFileName': 'Average of Averages', 'AverageGain': overall_avg_gain}, ignore_index=True)

        # Save the average gains DataFrame to a CSV file
        gains_file_path = os.path.join(directory_path, "average_gains.csv")
        gains_df.to_csv(gains_file_path, index=False)
        print(f"Average gains saved to {gains_file_path}")
