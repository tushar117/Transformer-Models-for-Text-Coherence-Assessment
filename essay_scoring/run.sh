#!/bin/bash

# NOTE: essay scoring python module with use single GPU if its available.
# fetches the directory where the shell file resides
file_path=$(realpath "$0")
dir_path=$(dirname "$file_path")

#setting up the defaults
START_INDEX=1 #start prompt id
OFFSET=8 # controls maximum number of runs
CHECKPOINT_PATH=$dir_path/checkpoint   #change required
MODEL_DIR=$dir_path   #optional
PYTHON=$(which python)   #change required

RANK_BATCH_SIZE=1
SCORE_BATCH_SIZE=2
RANK_EPOCHS=10
SCORE_EPOCHS=10
RANK_LR=$(python -c 'print(1e-6)')  #CAUTION : BASH doesn't works with floating point values so using workaround.
SCORE_LR=$(python -c 'print(1e-5)')  #CAUTION : BASH doesn't works with floating point values so using workaround.

# seq length related configuration
MAX_SEQ_LENGTH=2048   # set for essay scoring dataset
MAX_FACT_COUNT=50
MAX_FACT_SEQ_LEN=50

#transformer model to use
ARCH='vanilla'  # can take values from ['vanilla', 'mtl', 'combined', 'fact-aware', 'hierarchical']
TF1_MODEL_NAME='allenai/longformer-base-4096'
TF2_MODEL_NAME='roberta-base'
PRETRAINED_TF2=0  #load pretrained weight for TF2 when greater than 0
DISABLE_MTL=0 # deactivates the MTL if "combined" architecture is selected
SENTENCE_POOLING='max' # if hierarchical and combined model is selected then one can choose from ['sum', 'mean', 'max', 'min', 'attention', 'none']

ONLINE_SYNC=1  #control w&b online syncronization, 0 means inactive
MTL_BASE_ARCH='vanilla' # can take values from ['vanilla', 'fact-aware', 'hierarchical']
DATASET_DIR=$MODEL_DIR
NSAMPLES=5

printf "\n\n"
#dynamically set above default values through shell arguments
while [ $# -gt 0 ]; do
  case "$1" in
    --start_index=*)
      START_INDEX="${1#*=}"
      ;;
    --offset=*)
      OFFSET="${1#*=}"
      ;;
    --checkpoint_path=*)
      CHECKPOINT_PATH="${1#*=}"/checkpoint
      ;;
    --model_dir=*)
      MODEL_DIR="${1#*=}"
      ;;
    --python=*)
      PYTHON="${1#*=}"
      ;;
    --rank_batch_size=*)
      RANK_BATCH_SIZE="${1#*=}"
      ;;
    --score_batch_size=*)
      SCORE_BATCH_SIZE="${1#*=}"
      ;;
    --rank_epochs=*)
      RANK_EPOCHS="${1#*=}"
      ;;
    --score_epochs=*)
      SCORE_EPOCHS="${1#*=}"
      ;;
    --rank_lr=*)
      RANK_LR="${1#*=}"
      ;;
    --score_lr=*)
      SCORE_LR="${1#*=}"
      ;;
    --max_seq_len=*)
      MAX_SEQ_LENGTH="${1#*=}"
      ;;
    --max_fact_count=*)
      MAX_FACT_COUNT="${1#*=}"
      ;;
    --max_fact_seq_len=*)
      MAX_FACT_SEQ_LEN="${1#*=}"
      ;;
    --arch=*)
      ARCH="${1#*=}"
      ;;
    --tf1_model=*)
      TF1_MODEL_NAME="${1#*=}"
      ;;
    --tf2_model=*)
      TF2_MODEL_NAME="${1#*=}"
      ;;
    --pretrained_tf2=*)
      PRETRAINED_TF2="${1#*=}"
      ;;
    --disable_mtl=*)
      DISABLE_MTL="${1#*=}"
      ;;
    --pool=*)
      SENTENCE_POOLING="${1#*=}"
      ;;
    --dataset_dir=*)
      DATASET_DIR="${1#*=}"
      ;;
    --mtl_base_arch=*)
      MTL_BASE_ARCH="${1#*=}"
      ;;
    --online=*)
      ONLINE_SYNC="${1#*=}"
      ;;
    --nsamples=*)
      NSAMPLES="${1#*=}"
      ;;
    *)
      printf "***************************\n"
      printf "* Error: Invalid argument. please check argument $1 *\n"
      printf "***************************\n"
      exit 1
  esac
  shift
done

# api key weight & Biases
# export WANDB_API_KEY=

# calculating the end index
END_INDEX=$( expr $START_INDEX + $OFFSET - 1 )

ARCH_NAME=$ARCH
if [ "$ARCH" = "mtl" ]; then
  ARCH_NAME="$ARCH-$MTL_BASE_ARCH"
fi

FULL_MODEL_NAME=$(echo "$ARCH_NAME-$TF1_MODEL_NAME")

# replacing os path separator if exists in full model name
FULL_MODEL_NAME=$($PYTHON -c "print('$FULL_MODEL_NAME'.replace('/', '-'))")

download_scipt_dir=$($PYTHON -c "from pathlib import Path; print(Path('$MODEL_DIR').parent.absolute())")
DATASET_DIR="$DATASET_DIR/processed_data-$FULL_MODEL_NAME"

#########################################################
#print argument captures in shell script
echo "<< ----------- Experiment configurations -------------"
echo "START_INDEX : $START_INDEX"
echo "OFFSET : $OFFSET"
echo "END_INDEX : $END_INDEX" 
echo "CHECKPOINT_PATH : $CHECKPOINT_PATH"
echo "MODEL_DIR : $MODEL_DIR"
echo "PYTHON : $PYTHON"
echo "RANK_BATCH_SIZE : $RANK_BATCH_SIZE"
echo "SCORE_BATCH_SIZE : $SCORE_BATCH_SIZE"
echo "RANK_EPOCHS : $RANK_EPOCHS"
echo "SCORE_EPOCHS : $SCORE_EPOCHS"
echo "RANK_LR : $RANK_LR"
echo "SCORE_LR : $SCORE_LR"
echo "MAX_SEQ_LENGTH : $MAX_SEQ_LENGTH"
echo "MAX_FACT_COUNT : $MAX_FACT_COUNT"
echo "MAX_FACT_SEQ_LEN : $MAX_FACT_SEQ_LEN"
echo "ARCH : $ARCH"
echo "TF1_MODEL_NAME : $TF1_MODEL_NAME"
echo "TF2_MODEL_NAME : $TF2_MODEL_NAME"
echo "PRETRAINED_TF2 : $PRETRAINED_TF2"
echo "DISABLE_MTL : $DISABLE_MTL"
echo "SENTENCE POOLING : $SENTENCE_POOLING"
echo "ONLINE_SYNC : $ONLINE_SYNC"
echo "DATASET_DIR : $DATASET_DIR"
if [ "$ARCH" = "mtl" ]; then
  echo "MTL_BASE_ARCH : $MTL_BASE_ARCH"
fi
echo "NSAMPLES: $NSAMPLES"
echo "--------------------------------------------------- >>"
printf "\n"

#create new checkpoint path
if [ ! -e $CHECKPOINT_PATH ]; then
    mkdir -p $CHECKPOINT_PATH
fi

#find the latest checkpoint in the directory
function get_checkpoint_file(){
    file_list=()
    for the_file in $CHECKPOINT_PATH/$1/*
    do
        if [[ -f $the_file ]]; then
            file_list+=($the_file)
        fi
    done

    if [[ ${#file_list[@]} -eq 1 ]]; then
        echo "${file_list[0]}"
    else
        printf "exiting as more than one file present in checkpoint path : $CHECKPOINT_PATH/$1 : ${file_list[@]}"
        exit 1
    fi
}

#primary execution loop for model training and inference over $MAX_RUNS iterations 
for i in $(seq $START_INDEX $END_INDEX)
do
    CUR_MODEL_NAME=$(echo "$ARCH-$TF1_MODEL_NAME-prompt-$i")
    # delete alreay existing directory and download the processed dataset
    local_dataset_dir=$(echo "$DATASET_DIR-prompt-$i")
    $PYTHON $download_scipt_dir/download_data.py --store_path $local_dataset_dir
    # remove previously generated checkpoints
    if [ -e $CHECKPOINT_PATH/$CUR_MODEL_NAME ]; then
      echo "removing previously obtained checkpoints"
      rm -rf $CHECKPOINT_PATH/$CUR_MODEL_NAME/*
    fi
    echo "execution iteration on prompt : #$i for model $FULL_MODEL_NAME"
    $PYTHON $MODEL_DIR/main.py --arch $ARCH --mtl_base_arch $MTL_BASE_ARCH --prompt_id $i --processed_dataset_path $local_dataset_dir --rank_epochs $RANK_EPOCHS --score_epochs $SCORE_EPOCHS --rank_batch_size $RANK_BATCH_SIZE --score_batch_size $SCORE_BATCH_SIZE --max_seq_len $MAX_SEQ_LENGTH --checkpoint_path $CHECKPOINT_PATH --model_name $TF1_MODEL_NAME --tf2_model_name $TF2_MODEL_NAME --freeze_emb_layer --disable_mtl $DISABLE_MTL --online_mode $ONLINE_SYNC --max_fact_count $MAX_FACT_COUNT --max_fact_seq_len $MAX_FACT_SEQ_LEN --sentence_pooling $SENTENCE_POOLING --use_pretrained_tf2 $PRETRAINED_TF2 --nsamples $NSAMPLES --margin 0.1 --score_clip_norm 1.0 --score_weight_decay 0.1
    #clean the checkpoint directory
    rm -rf $CHECKPOINT_PATH/$CUR_MODEL_NAME/*
    rm -rf $local_dataset_dir
done
