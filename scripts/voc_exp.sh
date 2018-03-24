
CFG="BS400-W620-S1200-case-bidir_test"

WORD2VEC_MODEL="/mnt/brain2/scratch/llajan/word2vec/GoogleNews-vectors-negative300.bin"

RESULTS_HOME="/mnt/brain6/scratch/llajan/s2v_results"
CHECKPOINT_PATH="$RESULTS_HOME/$CFG/train"
EXP_VOCAB_DIR="$RESULTS_HOME/$CFG/exp_vocab"

DATASET='BC'
#DATASET='BC_UMBC'
DATA_DIR="/home/llajan/b6/s2v_data"
VOCAB="$DATA_DIR/$DATASET/dictionary.txt"

export CUDA_VISIBLE_DEVICES=0
for INOUT in "" "_out"
do
python src/vocabulary_expansion.py \
  --model=${CHECKPOINT_PATH} \
  --model_vocab=${VOCAB} \
  --word2vec_model=${WORD2VEC_MODEL} \
  --output_dir=${EXP_VOCAB_DIR} \
  --inout=${INOUT}
done
