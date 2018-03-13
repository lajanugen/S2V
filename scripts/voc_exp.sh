
DIR="BS400-W620-S800-case-bidir"

RESULTS_HOME="/mnt/brain6/scratch/llajan/s2v_results"
CHECKPOINT_PATH="$RESULTS_HOME/$DIR/train"
SKIP_THOUGHTS_VOCAB="sent2vec/dictionaries/BookCorpus_freq_case_50k"
WORD2VEC_MODEL="/mnt/brain2/scratch/llajan/word2vec/GoogleNews-vectors-negative300.bin"
EXP_VOCAB_DIR="$RESULTS_HOME/$DIR/exp_vocab"
INOUT=""
#INOUT="_out"

export CUDA_VISIBLE_DEVICES=0
python sent2vec/vocabulary_expansion.py \
  --skip_thoughts_model=${CHECKPOINT_PATH} \
  --skip_thoughts_vocab=${SKIP_THOUGHTS_VOCAB} \
  --word2vec_model=${WORD2VEC_MODEL} \
  --output_dir=${EXP_VOCAB_DIR} \
  --inout=${INOUT}
