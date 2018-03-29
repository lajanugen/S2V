
TASK='SICK'
#TASK='TREC'
#TASK='MSRP'
#TASK='MR'
#TASK='CR'
#TASK='SUBJ'
#TASK='MPQA'

MDLS_PATH="s2v_models"
MDL_CFGS="model_configs"
GLOVE_PATH="dictionaries/GloVe"

#CFG="BS400-W620-S1200-case-bidir"
CFG="MC-BC"
#CFG="MC-UMBC"

SKIPTHOUGHTS="ST_dir"
DATA="ST_data"

export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="$SKIPTHOUGHTS:$PYTHONPATH"
python src/evaluate.py \
	--eval_task=$TASK \
	--data_dir=$DATA \
	--model_config="$MDL_CFGS/$CFG/eval.json" \
	--results_path="$MDLS_PATH" \
	--Glove_path=$GLOVE_PATH

