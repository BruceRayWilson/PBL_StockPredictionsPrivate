    @staticmethod
    def train() -> None:
        '''Trains LLM, evaluates on a validation set, and prints evaluation metrics.'''
        print("\nLLM training and evaluation...")

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

        # Tokenizer updated to DistilBertTokenizerFast
        tokenizer = DistilBertTokenizerFast.from_pretrained(model_name)

        # Updated preprocess_function to use tokenizer.__call__ method for better performance
        def preprocess_function(examples):
            return tokenizer(examples[text_column_name], truncation=True, padding=True)

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
