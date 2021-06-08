#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA_ROOT=$PROJECT_ROOT/dataset/email_search/

# model
MODEL_TYPE=bert
MODEL_NAME=bert-base-multilingual-cased
#export MODEL_NAME=/home/t-yaqwan/Projects/source_code/language_model/output_all_query
MODEL_NAME=bert-base-multilingual-cased
#MODEL_TYPE=roberta
#MODEL_NAME=roberta-base
# params
LR=5e-5
WEIGHT_DECAY=5e-6
EPOCH=15
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=20

TRAIN_BATCH=16
EVAL_BATCH=32
export MAX_LENGTH=64
export BERT_MODEL=bert-base-multilingual-cased



# self-training parameters
REINIT=0
BEGIN_STEP=100
LABEL_MODE=soft
PERIOD=700
HP_LABEL=5.9
SHOT=10
SCENARIO=mobile

#python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/train_whole.txt $BERT_MODEL $MAX_LENGTH > train.txt
#python3 preprocess.py ./train_aug_back.txt $BERT_MODEL $MAX_LENGTH > train.txt
python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/train_whole_${SHOT}.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/train.txt
python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/unlabeled_train_whole_${SHOT}.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/unlabeled_train.txt
#python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/aug_train_whole.txt $BERT_MODEL $MAX_LENGTH > train.txt
python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/dev_whole.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/dev.txt
python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/validate_bio/whole_${SCENARIO}.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/test.txt
#python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/train_whole_30.txt $BERT_MODEL $MAX_LENGTH > test.txt
#python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/train_whole_70.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/test.txt
#preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/aug_train_whole.txt $BERT_MODEL $MAX_LENGTH > test.txt
#python3 preprocess.py /home/t-yaqwan/Projects/data/EmailSearch/train_bio/aug_train_whole.txt $BERT_MODEL $MAX_LENGTH > test.txt
cat $DATA_ROOT/train.txt $DATA_ROOT/unlabeled_train.txt $DATA_ROOT/dev.txt $DATA_ROOT/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_ROOT/labels.txt

# output
OUTPUT=$PROJECT_ROOT/outputs/email_search/self_training/${SCENARIO}_${SHOT}/${MODEL_TYPE}_reinit${REINIT}_begin${BEGIN_STEP}_period${PERIOD}_${LABEL_MODE}_hp${HP_LABEL}_${EPOCH}_${LR}/

[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'dataset' --exclude 'pretrained_model' --exclude 'outputs' $PROJECT_ROOT/ $OUTPUT/src

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 run_self_training_ner.py --data_dir $DATA_ROOT \
  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --logging_steps 100 \
  --save_steps 100000 \
  --do_predict \
  --overwrite_cache \
  --evaluate_during_training \
  --output_dir $OUTPUT \
  --cache_dir $PROJECT_ROOT/pretrained_model \
  --seed $SEED \
  --max_seq_length $MAX_LENGTH \
  --overwrite_output_dir \
  --self_training_reinit $REINIT --self_training_begin_step $BEGIN_STEP --self_training_period $PERIOD \
  --self_training_label_mode $LABEL_MODE --self_training_hp_label $HP_LABEL \
  --use_product