
NUM_WORDS=50001
OUTPUT_DIR="TFRecords" 
VOCAB_FILE="dictionary.txt"  
TOKENIZED_FILES="BookCorpus/txt_tokenized/*"

python src/data/preprocess_dataset.py \
  --input_files "$TOKENIZED_FILES" \
  --vocab_file $VOCAB_FILE \
  --output_dir $OUTPUT_DIR \
  --num_words $NUM_WORDS \
  --max_sentence_length 50 \
  --case_sensitive
