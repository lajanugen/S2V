#TASK='TREC'
TASK='SICK'
#TASK='MSRP'
#TASK='MR'
#TASK='CR'
#TASK='SUBJ'
#TASK='MPQA'

MDLS_PATH="/home/llajan/b6/s2v_models/"
MDL_CFGS="/home/llajan/b6/s2v_model_configs/"

CFG="MC-BC"

export CUDA_VISIBLE_DEVICES=0
python src/evaluate.py \
	--eval_task=$TASK \
	--data_dir="/home/llajan/sent2vec/skipthoughts/data" \
	--model_config="$MDL_CFGS/$CFG/eval.json" \
	--results_path="$MDLS_PATH"

