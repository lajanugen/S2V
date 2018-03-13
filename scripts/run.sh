#DATA="TFRecords"
#DATA="TFRecords_Glove"

DATA_DIR="/home/llajan/b6/s2v_data"

RESULTS_HOME="/mnt/brain6/scratch/llajan/s2v_results"

#DIR="BS400-W620-S1200-case-bidir"
#DIR="BS400-W300-bow-Glove-BC-ctxt3-drop-noise1k"


CFG=$DIR

DIR="$RESULTS_HOME/$DIR"

DATASET='BC'
NUM_INST=45786400
VOCAB_SIZE=50001

DATASET='BC-UMBC'
NUM_INST=174817800
VOCAB_SIZE=100000

BS=400
SEQ_LEN=30

export CUDA_VISIBLE_DEVICES=0
python sent2vec/train.py \
    --input_file_pattern="$DATA_DIR/$DATASET/$DATA/train-?????-of-00100" \
    --train_dir="$DIR/train" \
    --learning_rate_decay_factor=0 \
    --batch_size=$BS \
    --sequence_length=$SEQ_LEN \
    --nepochs=1 \
    --num_train_inst=$NUM_INST \
    --save_model_secs=1800 \
    --model_config="model_configs/$CFG/train.json" &


export CUDA_VISIBLE_DEVICES=3
python sent2vec/eval.py \
    --input_file_pattern="$DATA_DIR/$DATASET/$DATA/validation-?????-of-00001" \
    --checkpoint_dir="$DIR/train" \
    --eval_dir="$DIR/eval" \
    --batch_size=$BS \
    --model_config="model_configs/$CFG/train.json" \
    --eval_interval_secs=1800 \
    --sequence_length=$SEQ_LEN &
