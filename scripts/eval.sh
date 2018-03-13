#TASK='TREC'
TASK='SICK'
#TASK='MSRP'
#TASK='MR'
#TASK='CR'
#TASK='SUBJ'
#TASK='MPQA'

MDLS_PATH="/home/llajan/b6/s2v_models/"
MDL_CFGS="/home/llajan/b6/s2v_model_configs/"
GLOVE_PATH="/home/llajan/sent2vec/sent2vec/dictionaries/glove.840B.300d.txt"

#CFG="MC-BC"
CFG="MC-UMBC"

export CUDA_VISIBLE_DEVICES=0
python src/evaluate.py \
	--eval_task=$TASK \
	--data_dir="/home/llajan/sent2vec/skipthoughts/data" \
	--model_config="$MDL_CFGS/$CFG/eval.json" \
	--results_path="$MDLS_PATH" \
	--Glove_path=$GLOVE_PATH

