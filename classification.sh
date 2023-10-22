dataset="master_train_data/master.csv"
test_dataset="test_data/test.csv"
model=bert-base-uncased
python ~/dl/github.com/huggingface/transformers/examples/pytorch/text-classification/run_classification.py \
    --model_name_or_path  ${model} \
    --train_file ${dataset} \
    --validation_file ${test_dataset} \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "Sentence" \
    --text_column_delimiter "," \
    --label_column_name "Gain" \
    --do_train \
    --do_eval \
    --max_seq_length 64 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir Test_003/
    