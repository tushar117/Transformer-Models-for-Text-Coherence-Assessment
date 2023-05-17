#!/bin/bash

# fetches the directory where the shell file resides
file_path=$(realpath "$0")
dir_path=$(dirname "$file_path")

#setting up the defaults
START_INDEX=1
OFFSET=10 # controls maximum number of runs
GPUS=1
CHECKPOINT_PATH=$dir_path/checkpoint   #change required
MODEL_DIR=$dir_path   #optional
PYTHON=$(which python)   #change required

BATCH_SIZE=1
TEST_BATCH_SIZE=2
EPOCHS=10
LR=$(python -c 'print(1e-6)')  #CAUTION : BASH doesn't works with floating point values so using workaround.

# seq length related configuration
MAX_SEQ_LENGTH=512   # set 2048 for wsj dataset
MAX_FACT_COUNT=50
MAX_FACT_SEQ_LEN=50

#transformer model to use
ARCH='vanilla'  # can take values from ['vanilla', 'mtl', 'combined', 'fact-aware', 'hierarchical']
TF1_MODEL_NAME='roberta-base'
TF2_MODEL_NAME='roberta-base'
PRETRAINED_TF2=0  #load pretrained weight for TF2 when greater than 0
DISABLE_MTL=0 # deactivates the MTL if "combined" architecture is selected
SENTENCE_POOLING='none' # if hierarchical and combined model is selected then one can choose from ['sum', 'mean', 'max', 'min', 'attention', 'none']

# corpus related congiguration
CORPUS='gcdc' # can take values from ['wsj', 'gcdc']
SUB_CORPUS='Yahoo' # can take values from ['All', 'Yahoo', 'Clinton', 'Enron', 'Yelp'] if 'gcdc' is selected as $CORPUS
PERMUTATIONS=20
WITH_REPLACEMENT=1   #if greater than zero it will draw sample with replacement otherwise without replacement
TASK='3-way-classification'  # can take values from ['3-way-classification', 'minority-classification', 'sentence-ordering', 'sentence-score-prediction']

MIXED_PRECISION=0
ONLINE_SYNC=0  #control w&b online syncronization, 0 means inactive

MTL_BASE_ARCH='vanilla' # can take values from ['vanilla', 'fact-aware', 'hierarchical']
DATASET_DIR=$MODEL_DIR

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
    --gpus=*)
      GPUS="${1#*=}"
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
    --batch_size=*)
      BATCH_SIZE="${1#*=}"
      TEST_BATCH_SIZE=$BATCH_SIZE
      ;;
    --test_batch_size=*)
      TEST_BATCH_SIZE="${1#*=}"
      ;;
    --epochs=*)
      EPOCHS="${1#*=}"
      ;;
    --lr=*)
      LR="${1#*=}"
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
    --corpus=*)
      CORPUS="${1#*=}"
      ;;
    --sub_corpus=*)
      SUB_CORPUS="${1#*=}"
      ;;  
    --permutations=*)
      PERMUTATIONS="${1#*=}"
      ;;
    --with_replacement=*)
      WITH_REPLACEMENT="${1#*=}"
      ;;
    --online=*)
      ONLINE_SYNC="${1#*=}"
      ;;
    --task=*)
      TASK="${1#*=}"
      ;;
    --mixed_precision=*)
      MIXED_PRECISION="${1#*=}"
      ;;
    --mtl_base_arch=*)
      MTL_BASE_ARCH="${1#*=}"
      ;;
    --dataset_dir=*)
      DATASET_DIR="${1#*=}"
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

ARCH_NAME=$ARCH
if [ "$ARCH" = "mtl" ]; then
  ARCH_NAME="$ARCH-$MTL_BASE_ARCH"
fi

if [ "$CORPUS" = "wsj" ]; then
  FULL_MODEL_NAME=$(echo "$CORPUS-$ARCH_NAME-$TASK-$TF1_MODEL_NAME")
else
  FULL_MODEL_NAME=$(echo "$CORPUS-$SUB_CORPUS-$ARCH_NAME-$TASK-$TF1_MODEL_NAME")
fi

# replacing os path separator if exists in full model name
FULL_MODEL_NAME=$($PYTHON -c "print('$FULL_MODEL_NAME'.replace('/', '-'))")

# calculating the end index
END_INDEX=$( expr $START_INDEX + $OFFSET - 1 )
# using different processed directory for different process
DATASET_DIR="$DATASET_DIR/processed_data-$FULL_MODEL_NAME"

#########################################################
#print argument captures in shell script
echo "<< ----------- Experiment configurations -------------"
echo "START_INDEX : $START_INDEX"
echo "OFFSET : $OFFSET"
echo "END_INDEX : $END_INDEX" 
echo "GPUS : $GPUS"
echo "CHECKPOINT_PATH : $CHECKPOINT_PATH"
echo "MODEL_DIR : $MODEL_DIR"
echo "PYTHON : $PYTHON"
echo "BATCH_SIZE : $BATCH_SIZE"
echo "TEST_BATCH_SIZE : $TEST_BATCH_SIZE"
echo "EPOCHS : $EPOCHS"
echo "LR : $LR"
echo "MAX_SEQ_LENGTH : $MAX_SEQ_LENGTH"
echo "MAX_FACT_COUNT : $MAX_FACT_COUNT"
echo "MAX_FACT_SEQ_LEN : $MAX_FACT_SEQ_LEN"
echo "ARCH : $ARCH"
echo "TF1_MODEL_NAME : $TF1_MODEL_NAME"
echo "TF2_MODEL_NAME : $TF2_MODEL_NAME"
echo "PRETRAINED_TF2 : $PRETRAINED_TF2"
echo "DISABLE_MTL : $DISABLE_MTL"
echo "SENTENCE POOLING : $SENTENCE_POOLING"
echo "CORPUS : $CORPUS"
if [ "$CORPUS" = "gcdc" ]; then
  echo "SUB_CORPUS : $SUB_CORPUS"
fi
echo "PERMUTATIONS : $PERMUTATIONS"
echo "WITH_REPLACEMENT : $WITH_REPLACEMENT"
echo "ONLINE_SYNC : $ONLINE_SYNC"
echo "TASK : $TASK"
echo "DATASET_DIR : $DATASET_DIR"
echo "MIXED PRECISION : $MIXED_PRECISION"
if [ "$ARCH" = "mtl" ]; then
  echo "MTL_BASE_ARCH : $MTL_BASE_ARCH"
fi
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

# only use when gcdc is selected
sub_corpuses=("Yahoo" "Clinton" "Enron" "Yelp")

#primary execution loop for model training and inference over $MAX_RUNS iterations 
for i in $(seq $START_INDEX $END_INDEX)
do
    local_dataset_dir=$(echo "$DATASET_DIR-$i")
    #Download processed dataset
    # $PYTHON $MODEL_DIR/download_data.py --store_path $local_dataset_dir
    CUR_MODEL_NAME=$FULL_MODEL_NAME-$i
    # remove previously generated checkpoints
    if [ -e $CHECKPOINT_PATH/$CUR_MODEL_NAME ]; then
      echo "removing previously obtained checkpoints"
      rm -rf $CHECKPOINT_PATH/$CUR_MODEL_NAME/*
    fi
    echo "train iteration #$i for model $FULL_MODEL_NAME"
    $PYTHON $MODEL_DIR/main.py --arch $ARCH --mtl_base_arch $MTL_BASE_ARCH --processed_dataset_path $local_dataset_dir --epochs $EPOCHS --gpus $GPUS --batch_size $BATCH_SIZE --max_seq_len $MAX_SEQ_LENGTH --checkpoint_path $CHECKPOINT_PATH --learning_rate $LR --corpus $CORPUS --sub_corpus $SUB_CORPUS --model_name $TF1_MODEL_NAME --tf2_model_name $TF2_MODEL_NAME --freeze_emb_layer --exp_count $i --online_mode $ONLINE_SYNC --task $TASK --max_fact_count $MAX_FACT_COUNT --max_fact_seq_len $MAX_FACT_SEQ_LEN --disable_mtl $DISABLE_MTL --with_replacement $WITH_REPLACEMENT --fp16 $MIXED_PRECISION --sentence_pooling $SENTENCE_POOLING --use_pretrained_tf2 $PRETRAINED_TF2
    
    echo "test iteration #$i for model $FULL_MODEL_NAME" 
    # get the full path of model checkpoint for best performing model on validation dataset
    checkpoint_file=$(get_checkpoint_file $CUR_MODEL_NAME)
    echo "using checkpoint file for model $CUR_MODEL_NAME : $checkpoint_file"
    
    # test the model
    if [ "$CORPUS" = "gcdc" ]; then
      for sub_corpus in ${sub_corpuses[@]}
      do
          # on gcdc sub_corpus
          $PYTHON $MODEL_DIR/main.py --arch $ARCH --mtl_base_arch $MTL_BASE_ARCH --processed_dataset_path $local_dataset_dir --inference --batch_size $TEST_BATCH_SIZE --max_seq_len $MAX_SEQ_LENGTH --checkpoint_path $checkpoint_file --corpus $CORPUS --sub_corpus $sub_corpus --model_name $TF1_MODEL_NAME --tf2_model_name $TF2_MODEL_NAME --freeze_emb_layer --exp_count $i --online_mode $ONLINE_SYNC --task $TASK --max_fact_count $MAX_FACT_COUNT --max_fact_seq_len $MAX_FACT_SEQ_LEN --with_replacement $WITH_REPLACEMENT --fp16 $MIXED_PRECISION --logger_exp_name $FULL_MODEL_NAME-$sub_corpus-$i --gpus $GPUS --sentence_pooling $SENTENCE_POOLING --use_pretrained_tf2 $PRETRAINED_TF2
      done
    else
      # on wsj test corpus
      $PYTHON $MODEL_DIR/main.py --arch $ARCH --mtl_base_arch $MTL_BASE_ARCH --processed_dataset_path $local_dataset_dir --inference --batch_size $TEST_BATCH_SIZE --max_seq_len $MAX_SEQ_LENGTH --checkpoint_path $checkpoint_file --corpus $CORPUS --model_name $TF1_MODEL_NAME --tf2_model_name $TF2_MODEL_NAME --freeze_emb_layer --exp_count $i --online_mode $ONLINE_SYNC --task $TASK --max_fact_count $MAX_FACT_COUNT --max_fact_seq_len $MAX_FACT_SEQ_LEN --with_replacement $WITH_REPLACEMENT --fp16 $MIXED_PRECISION --gpus $GPUS --logger_exp_name $CUR_MODEL_NAME --sentence_pooling $SENTENCE_POOLING --use_pretrained_tf2 $PRETRAINED_TF2
    fi
    #clean the checkpoint directory
    rm -rf $CHECKPOINT_PATH/$CUR_MODEL_NAME/*
    rm -rf $local_dataset_dir
done

# finally clean the dataset
