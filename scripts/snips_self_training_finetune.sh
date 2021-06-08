#!/bin/bash

if [[ $# -ne 1 ]]; then
  GPUID=0
else
  GPUID=$1
fi

echo "Run on GPU $GPUID"

# data
PROJECT_ROOT=$(dirname "$(readlink -f "$0")")/..
DATA_ROOT=$PROJECT_ROOT/dataset/snips/

# model
MODEL_TYPE=bert
#MODEL_NAME=roberta-base
#export MODEL_NAME=/home/t-yaqwan/Projects/source_code/language_model/output_all_query

#MODEL_TYPE=roberta
#MODEL_NAME=roberta-base
# params
LR=5e-5
CONTROLER_LR=0.001
WEIGHT_DECAY=5e-6
EPOCH=70
SEED=0

ADAM_EPS=1e-8
ADAM_BETA1=0.9
ADAM_BETA2=0.98
WARMUP=20

USE_CONTROLER=weight


MT_ALPHA1=0.995
MT_ALPHA2=0.99

TRAIN_BATCH=16
UN_TRAIN_BATCH=8
EVAL_BATCH=32
export MAX_LENGTH=64
export BERT_MODEL=bert-base-multilingual-cased
MODEL_NAME=bert-base-uncased



# self-training parameters
REINIT=1
BEGIN_STEP=200
LABEL_MODE=hard
PERIOD=2000
HP_LABEL=5.0
SHOT=20
SCENARIO=mobile

python3 preprocess.py $DATA_ROOT/train_whole_${SHOT}.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/train.txt
python3 preprocess.py $DATA_ROOT/unlabeled_train_whole_${SHOT}.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/unlabeled_train.txt
python3 preprocess.py $DATA_ROOT/valid_whole.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/dev.txt
python3 preprocess.py $DATA_ROOT/test_whole.txt $BERT_MODEL $MAX_LENGTH > $DATA_ROOT/test.txt
cat $DATA_ROOT/train.txt $DATA_ROOT/unlabeled_train.txt $DATA_ROOT/dev.txt $DATA_ROOT/test.txt | cut -d " " -f 2 | grep -v "^$"| sort | uniq > $DATA_ROOT/labels.txt

# output
OUTPUT=$PROJECT_ROOT/outputs/snips/self_training_controler/${SHOT}/${MODEL_TYPE}_reinit${REINIT}_begin${BEGIN_STEP}_period${PERIOD}_${LABEL_MODE}_hp${HP_LABEL}_${EPOCH}_${LR}_${USE_CONTROLER}/

[ -e $OUTPUT/script  ] || mkdir -p $OUTPUT/script
cp -f $(readlink -f "$0") $OUTPUT/script
rsync -ruzC --exclude-from=$PROJECT_ROOT/.gitignore --exclude 'dataset' --exclude 'pretrained_model' --exclude 'outputs' $PROJECT_ROOT/ $OUTPUT/src

CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$GPUID python3 run_self_training_teacher_finetune.py --data_dir $DATA_ROOT \
  --model_type $MODEL_TYPE --model_name_or_path $MODEL_NAME \
  --learning_rate $LR \
  --weight_decay $WEIGHT_DECAY \
  --adam_epsilon $ADAM_EPS \
  --adam_beta1 $ADAM_BETA1 \
  --adam_beta2 $ADAM_BETA2 \
  --num_train_epochs $EPOCH \
  --warmup_steps $WARMUP \
  --per_gpu_train_batch_size $TRAIN_BATCH \
  --per_pgu_labeled_batch_size 4 \
  --per_gpu_unlabeled_train_batch_size $UN_TRAIN_BATCH \
  --per_gpu_eval_batch_size $EVAL_BATCH \
  --logging_steps 50 \
  --save_steps 1000 \
  --do_train \
  --do_eval \
  --do_predict \
  --overwrite_cache \
  --evaluate_during_training \
  --output_dir $OUTPUT \
  --cache_dir $PROJECT_ROOT/pretrained_model \
  --mt_alpha1 $MT_ALPHA1 \
  --mt_alpha2 $MT_ALPHA2  \
  --seed $SEED \
  --max_seq_length $MAX_LENGTH \
  --overwrite_output_dir \
  --self_training_reinit $REINIT --self_training_begin_step $BEGIN_STEP --self_training_period $PERIOD \
  --self_training_label_mode $LABEL_MODE --self_training_hp_label $HP_LABEL \
  --controler_step_size $CONTROLER_LR \
  --use_controler \
  --controler_sampling_steps -1 \
  --per_gpu_meta_batch_size 8 \
  --labeled_beta 1 \
  --labeled_beta_decay 0.9\
  --use_labeled_loss \
  --max_steps 5000 \
  --use_MPL 1 \
  --teacher_mpl_start_steps 10 \
  --use_focal_loss
