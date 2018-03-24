# Quick-Thought Vectors

This is a TensorFlow implementation accompanying our paper

Lajanugen Logeswaran, Honglak Lee. 
An efficient framework for learning sentence representations. In ICLR, 2018.

### Contents
#* [Model Overview](#model-overview)
#* [Getting Started](#getting-started)
#    * [Install Required Packages](#install-required-packages)
#    * [Download Pretrained Models (Optional)](#download-pretrained-models-optional)
#* [Training a Model](#training-a-model)
#    * [Prepare the Training Data](#prepare-the-training-data)
#    * [Run the Training Script](#run-the-training-script)
#* [Expanding the Vocabulary](#expanding-the-vocabulary)
#    * [Overview](#overview)
#    * [Preparation](#preparation)
#    * [Run the Vocabulary Expansion Script](#run-the-vocabulary-expansion-script)
#* [Evaluating a Model](#evaluating-a-model)
#    * [Overview](#overview-1)
#    * [Preparation](#preparation-1)
#    * [Run the Evaluation Tasks](#run-the-evaluation-tasks)
#* [Encoding Sentences](#encoding-sentences)


### Download Pretrained Models (Optional)

You can download model checkpoints pretrained on the
[BookCorpus](http://yknzhu.wixsite.com/mbweb) dataset in the following
configurations:

Pre-trained models can be downloaded from [http://bit.ly/2ptSYXT](http://bit.ly/2ptSYXT).

* Uni/Bi/Combine-QT
* MC-QT

## Model configuration files

We use json configuration files to describe models. These configuration files provide a concise description of a model. They also make it easy to concatenate representations from different models/types of models at evaluation time.

The description of a sentence encoder looks like the following.
```
{
	"encoder": "gru",				# Type of encoder
	"encoder_dim": 1200,				# Dimensionality of encoder
	"bidir": true,					# Uni/bi directional
	"checkpoint_path": "",				# Path to checkpoint
	"vocab_configs": [				# Configuration of vocabulary/word embeddings
	{
		"mode": "trained",			# Vocabulary mode: fixed/trained/expand
		"name": "word_embedding",
		"dim": 620,				# Word embedding size
		"size": 50001,				# Size of vocabulary
		"vocab_file": "BC_dictionary.txt",	# Dictionary file
		"embs_file": ""				# Provide external embeddings file
	}
	]
}
```

Vocabulary mode can be one of 'fixed', 'trained' or 'expand'. The 'fixed' mode represents the case where fixed, pre-trained (GloVe) embeddings are used. In the 'trained' mode, word embeddings are trained from scratch. The 'expand' mode, which is only used during evaluation on downstream tasks, refers to using an expanded vocabulary.

`checkpoint_path` and `vocab_file` have to be specified only for evaluation.

For using a concatenated representation at evaluation time, the json file can be a list of multiple encoder specifications. See `model_configs/BC/eval.json` for an example. 

## Training a Model

### Prepare the Training Data

The training script requires data to be in (sharded) TFRecord format. 
`scripts/data_prep.sh` can be used to generate these files.
The script requires a dictionary file and comma-separated paths to files containing tokenized sentences.
The dictionary file should have a single word in each line.
Each file is expected to have a tokenized sentence in each line, in the same order as the source document. 

### Run the Training Script

Use the `run.sh` script to train a model. 
The following variables have to be specified.

* DATA\_DIR 	# Path to TFRecord files
* RESULTS\_HOME # Directory to store results
* CFG 		# Name of model configuration 
* MDL\_CFGS 	# Path to model configuration files
* GLOVE\_PATH 	# Path to GloVe dictionary and embeddings

Example configuration files are provided in the model\_configs folder.

## Expanding the Vocabulary

Once the model is trained, the vocabulary used for training can be optionally expanded to a larger vocabulary using the technique proposed by the SkipThought paper. The `voc_exp.sh` script can be used to perform expansion. Since Word2Vec embeddings are used for expansion, you will have to download the Word2Vec model. The script assumes that the gensim library is avalable on the system. 

## Evaluating a Model

Use the `eval.sh` script for evaluation. The following variables need to be set.

* CFG 		# Name of model configuration 
* TASK 		# Name of the task
* MDLS\_PATH	# Path to model files
* MDL\_CFGS 	# Path to model configuration files
* GLOVE\_PATH 	# Path to GloVe dictionary and embeddings
* SKIPTHOUGHTS  # Path to SkipThoughts implementation
* DATA  	# Data directory for downstream tasks

Evaluation scripts for the downstream tasks from the authors of the SkipThought model are used. These scripts train a linear layer on top of the sentence embeddings for each task. 
You will need to clone or download the [skip-thoughts GitHub repository](https://github.com/ryankiros/skip-thoughts) by [ryankiros](https://github.com/ryankiros).
Set the `DATA` variable to the directory containing data for the downstream tasks. See the above repository for further details regarding downloading the data.

### Run the Evaluation Tasks

