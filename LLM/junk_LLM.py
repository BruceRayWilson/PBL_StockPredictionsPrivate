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

    @staticmethod
    def train() -> None:
        '''Trains LLM, evaluates on a validation set, and prints evaluation metrics.'''
        print("LLM training and evaluation...")

        # Verify that 'results' is a directory and delete it if it exists
        if os.path.isdir("results"):
            shutil.rmtree("results")

        data_path = "master_train_data/master.csv" #@param {type:"string"}
        text_column_name = "Sentence" #@param {type:"string"}
        label_column_name = "Gain" #@param {type:"string"}

        model_name = "distilbert-base-uncased" #@param {type:"string"}
        test_size = 0.2 #@param {type:"number"}
        num_labels = 6 #@param {type:"number"}

        df = pd.read_csv(data_path)

        le = preprocessing.LabelEncoder()
        le.fit(df[label_column_name].tolist())
        df['label'] = le.transform(df[label_column_name].tolist())

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
            per_device_train_batch_size=512,
            per_device_eval_batch_size=512,
            num_train_epochs=10,
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
