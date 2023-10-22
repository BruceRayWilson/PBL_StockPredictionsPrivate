dataset="amazon_reviews_multi"
subset="en"
model=bert-base-uncased
python ~/dl/github.com/huggingface/transformers/examples/pytorch/text-classification/run_classification.py \
    --model_name_or_path  ${model} \
    --dataset_name ${dataset} \
    --dataset_config_name ${subset} \
    --shuffle_train_dataset \
    --metric_name accuracy \
    --text_column_name "review_title,review_body,product_category" \
    --text_column_delimiter "\n" \
    --label_column_name stars \
    --do_train \
    --do_eval \
    --max_seq_length 512 \
    --per_device_train_batch_size 32 \
    --learning_rate 2e-5 \
    --num_train_epochs 1 \
    --output_dir /tmp/${dataset}_${subset}/
    