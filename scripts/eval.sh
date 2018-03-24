#TASK='TREC'
TASK='SICK'
#TASK='MSRP'
#TASK='MR'
#TASK='CR'
#TASK='SUBJ'
#TASK='MPQA'

MDLS_PATH="/home/llajan/b6/s2v_models/"
MDL_CFGS="model_configs"
GLOVE_PATH="/home/llajan/b6/s2v_dictionaries/GloVe"

CFG="BS400-W620-S1200-case-bidir"
#CFG="MC-BC"
#CFG="MC-UMBC"

SKIPTHOUGHTS="/home/llajan/sent2vec"
DATA="/home/llajan/sent2vec/skipthoughts/data"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="/home/llajan/sent2vec:$PYTHONPATH"
python src/evaluate.py \
	--eval_task=$TASK \
	--data_dir=$DATA \
	--model_config="$MDL_CFGS/$CFG/eval_noexp.json" \
	--results_path="$MDLS_PATH" \
	--Glove_path=$GLOVE_PATH

