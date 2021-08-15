#!/bin/bash

GPUID=0
echo "Run on GPU $GPUID"

# data
TASK=MIT_restaurant
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA_ROOT=./dataset/${TASK}/
SHOT=10

# model
MODEL_TYPE=bert
# params
LR=5e-5
EPOCH=70
SEED=0

WARMUP=20
TRAIN_BATCH=16
UN_TRAIN_BATCH=32
EVAL_BATCH=32
export MAX_LENGTH=64
export BERT_MODEL=bert-base-uncased
MODEL_NAME=bert-base-uncased

# self-training parameters
REINIT=1
BEGIN_STEP=4000
LABEL_MODE=hard
PERIOD=3000

python3 ./src/preprocess.py $DATA_ROOT/train_whole_${SHOT}.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/train.txt
python3 ./src/preprocess.py $DATA_ROOT/unlabeled_train_whole_${SHOT}.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/unlabeled_train.txt
python3 ./src/preprocess.py $DATA_ROOT/valid_whole.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/dev.txt
python3 ./src/preprocess.py $DATA_ROOT/test_whole.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/test.txt
cat $DATA_ROOT/train.txt $DATA_ROOT/unlabeled_train.txt $DATA_ROOT/dev.txt $DATA_ROOT/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_ROOT/labels.txt

# output
OUTPUT=$PROJECT_ROOT/outputs/${TASK}/${SHOT}/${MODEL_TYPE}_reinit${REINIT}_begin${BEGIN_STEP}_period${PERIOD}_${LABEL_MODE}_${EPOCH}_${LR}/

[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'dataset' --exclude 'pretrained_model' --exclude 'outputs' $PROJECT_ROOT/ $OUTPUT/src

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 ./src/run.py --data_dir $DATA_ROOT \
  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
  --learning_rate $LR \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_pgu_labeled_batch_size 4 \
  --per_gpu_unlabeled_train_batch_size $UN_TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --logging_steps 200 \
  --save_steps 1000 \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_cache \
  --evaluate_during_training \
  --output_dir $OUTPUT \
  --cache_dir $PROJECT_ROOT/pretrained_model \
  --seed $SEED \
  --max_seq_length $MAX_LENGTH \
  --overwrite_output_dir \
  --self_training_reinit $REINIT \
  --self_training_begin_step $BEGIN_STEP \
  --self_training_period $PERIOD \
  --self_training_label_mode $LABEL_MODE  \
  --controler_step_size $LR \
  --use_psuedo_data_selection \
  --controler_sampling_steps -1 \
  --per_gpu_meta_batch_size 32 \
  --labeled_beta 1 \
  --labeled_beta_decay 0.9\
  --use_labeled_loss \
  --max_steps 30000 \
  --sampling_strategy loss_decay \
  --use_token_weight 1 \
  --smoothness_type max \



