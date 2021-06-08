

python ./src/distillation.py \
    --data_dir /home/t-yaqwan/Projects/source_code/token-classification \
    --train_dir /home/t-yaqwan/Projects/data/EmailSearch/train_bio/train_syth_whole.txt \
    --fine_train_dir /home/t-yaqwan/Projects/data/EmailSearch/train_bio/train_whole.txt \
    --valid_dir /home/t-yaqwan/Projects/data/EmailSearch/validate_bio/whole_outlook.txt \
    --test_generated_dir datasets/Teams_communication/generated_data/communication_message_generated_no_contact.txt \
    --test_generated_no_contact_dir datasets/Teams_communication/generated_data/communication_message_generated_contact.txt \
    --target_set_dir datasets/Teams_communication/Target_set_message_new_conll.txt \
    --teacher_model_path /home/t-yaqwan/Projects/source_code/token-classification/germeval-model/all_query_augment_data \
    --bert_model bert-base-multilingual-cased \
    --task_name ner \
    --output_dir ./output_outlook_more_data_finetune \
    --do_train \
    --do_eval  \
    --learning_rate 5e-5 \
    --num_train_epochs 50 \
    --warmup_proportion 0.01 \
    --max_seq_length 64 \
    --train_batch_size 32 \
    --temperature 1 \
    --alpha 1 \
    --beta 0  \
    --unsupervised_train_corpus /home/t-yaqwan/Projects/source_code/token-classification/train_aug.txt \


