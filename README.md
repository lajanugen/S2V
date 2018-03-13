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
#    * [Track Training Progress](#track-training-progress)
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

* Uni/Bi/Combine-QT
* MC-QT

```shell
# Directory to download the pretrained models to.
PRETRAINED_MODELS_DIR="${HOME}/skip_thoughts/pretrained/"

mkdir -p ${PRETRAINED_MODELS_DIR}
cd ${PRETRAINED_MODELS_DIR}

# Download and extract the unidirectional model.
wget "http://download.tensorflow.org/models/skip_thoughts_uni_2017_02_02.tar.gz"
tar -xvf skip_thoughts_uni_2017_02_02.tar.gz
rm skip_thoughts_uni_2017_02_02.tar.gz

# Download and extract the bidirectional model.
wget "http://download.tensorflow.org/models/skip_thoughts_bi_2017_02_16.tar.gz"
tar -xvf skip_thoughts_bi_2017_02_16.tar.gz
rm skip_thoughts_bi_2017_02_16.tar.gz
```

You can now skip to the sections [Evaluating a Model](#evaluating-a-model) and
[Encoding Sentences](#encoding-sentences).


## Training a Model

### Prepare the Training Data

To train a model you will need to provide training data in TFRecord format. The
TFRecord format consists of a set of sharded files containing serialized
`tf.Example` protocol buffers. Each `tf.Example` proto contains three
sentences:

  * `encode`: The sentence to encode.
  * `decode_pre`: The sentence preceding `encode` in the original text.
  * `decode_post`: The sentence following `encode` in the original text.

Each sentence is a list of words. During preprocessing, a dictionary is created
that assigns each word in the vocabulary to an integer-valued id. Each sentence
is encoded as a list of integer word ids in the `tf.Example` protos.

We have provided a script to preprocess any set of text-files into this format.
You may wish to use the [BookCorpus](http://yknzhu.wixsite.com/mbweb) dataset.
Note that the preprocessing script may take **12 hours** or more to complete
on this large dataset.

```shell
# Comma-separated list of globs matching the input input files. The format of
# the input files is assumed to be a list of newline-separated sentences, where
# each sentence is already tokenized.
INPUT_FILES="${HOME}/skip_thoughts/bookcorpus/*.txt"

# Location to save the preprocessed training and validation data.
DATA_DIR="${HOME}/skip_thoughts/data"

# Build the preprocessing script.
cd tensorflow-models/skip_thoughts
bazel build -c opt //skip_thoughts/data:preprocess_dataset

# Run the preprocessing script.
bazel-bin/skip_thoughts/data/preprocess_dataset \
  --input_files=${INPUT_FILES} \
  --output_dir=${DATA_DIR}
```

When the script finishes you will find 100 training files and 1 validation file
in `DATA_DIR`. The files will match the patterns `train-?????-of-00100` and
`validation-00000-of-00001` respectively.

The script will also produce a file named `vocab.txt`. The format of this file
is a list of newline-separated words where the word id is the corresponding 0-
based line index. Words are sorted by descending order of frequency in the input
data. Only the top 20,000 words are assigned unique ids; all other words are
assigned the "unknown id" of 1 in the processed data.

### Run the Training Script

Execute the following commands to start the training script. By default it will
run for 500k steps (around 9 days on a GeForce GTX 1080 GPU).

```shell
# Directory containing the preprocessed data.
DATA_DIR="${HOME}/skip_thoughts/data"

# Directory to save the model.
MODEL_DIR="${HOME}/skip_thoughts/model"

# Build the model.
cd tensorflow-models/skip_thoughts
bazel build -c opt //skip_thoughts/...

# Run the training script.
bazel-bin/skip_thoughts/train \
  --input_file_pattern="${DATA_DIR}/train-?????-of-00100" \
  --train_dir="${MODEL_DIR}/train"
```

### Track Training Progress

Optionally, you can run the `track_perplexity` script in a separate process.
This will log per-word perplexity on the validation set which allows training
progress to be monitored on
[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard).

Note that you may run out of memory if you run the this script on the same GPU
as the training script. You can set the environment variable
`CUDA_VISIBLE_DEVICES=""` to force the script to run on CPU. If it runs too
slowly on CPU, you can decrease the value of `--num_eval_examples`.

```shell
DATA_DIR="${HOME}/skip_thoughts/data"
MODEL_DIR="${HOME}/skip_thoughts/model"

# Ignore GPU devices (only necessary if your GPU is currently memory
# constrained, for example, by running the training script).
export CUDA_VISIBLE_DEVICES=""

# Run the evaluation script. This will run in a loop, periodically loading the
# latest model checkpoint file and computing evaluation metrics.
bazel-bin/skip_thoughts/track_perplexity \
  --input_file_pattern="${DATA_DIR}/validation-?????-of-00001" \
  --checkpoint_dir="${MODEL_DIR}/train" \
  --eval_dir="${MODEL_DIR}/val" \
  --num_eval_examples=50000
```

If you started the `track_perplexity` script, run a
[TensorBoard](https://www.tensorflow.org/get_started/summaries_and_tensorboard)
server in a separate process for real-time monitoring of training summaries and
validation perplexity.

```shell
MODEL_DIR="${HOME}/skip_thoughts/model"

# Run a TensorBoard server.
tensorboard --logdir="${MODEL_DIR}"
```

## Expanding the Vocabulary

### Overview

The vocabulary generated by the preprocessing script contains only 20,000 words
which is insufficient for many tasks. For example, a sentence from Wikipedia
might contain nouns that do not appear in this vocabulary.

A solution to this problem described in the
[Skip-Thought Vectors](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf)
paper is to learn a mapping that transfers word representations from one model to
another. This idea is based on the "Translation Matrix" method from the paper
[Exploiting Similarities Among Languages for Machine Translation](https://arxiv.org/abs/1309.4168).


Specifically, we will load the word embeddings from a trained *Skip-Thoughts*
model and from a trained [word2vec model](https://arxiv.org/pdf/1301.3781.pdf)
(which has a much larger vocabulary). We will train a linear regression model
without regularization to learn a linear mapping from the word2vec embedding
space to the *Skip-Thoughts* embedding space. We will then apply the linear
model to all words in the word2vec vocabulary, yielding vectors in the *Skip-
Thoughts* word embedding space for the union of the two vocabularies.

The linear regression task is to learn a parameter matrix *W* to minimize
*|| X - Y \* W ||<sup>2</sup>*, where *X* is a matrix of *Skip-Thoughts*
embeddings of shape `[num_words, dim1]`, *Y* is a matrix of word2vec embeddings
of shape `[num_words, dim2]`, and *W* is a matrix of shape `[dim2, dim1]`.

### Preparation

First you will need to download and unpack a pretrained
[word2vec model](https://arxiv.org/pdf/1301.3781.pdf) from
[this website](https://code.google.com/archive/p/word2vec/)
([direct download link](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)).
This model was trained on the Google News dataset (about 100 billion words).


Also ensure that you have already [installed gensim](https://radimrehurek.com/gensim/install.html).

### Run the Vocabulary Expansion Script

```shell
# Path to checkpoint file or a directory containing checkpoint files (the script
# will select the most recent).
CHECKPOINT_PATH="${HOME}/skip_thoughts/model/train"

# Vocabulary file generated by the preprocessing script.
SKIP_THOUGHTS_VOCAB="${HOME}/skip_thoughts/data/vocab.txt"

# Path to downloaded word2vec model.
WORD2VEC_MODEL="${HOME}/skip_thoughts/googlenews/GoogleNews-vectors-negative300.bin"

# Output directory.
EXP_VOCAB_DIR="${HOME}/skip_thoughts/exp_vocab"

# Build the vocabulary expansion script.
cd tensorflow-models/skip_thoughts
bazel build -c opt //skip_thoughts:vocabulary_expansion

# Run the vocabulary expansion script.
bazel-bin/skip_thoughts/vocabulary_expansion \
  --skip_thoughts_model=${CHECKPOINT_PATH} \
  --skip_thoughts_vocab=${SKIP_THOUGHTS_VOCAB} \
  --word2vec_model=${WORD2VEC_MODEL} \
  --output_dir=${EXP_VOCAB_DIR}
```

## Evaluating a Model

### Overview

The model can be evaluated using the benchmark tasks described in the
[Skip-Thought Vectors](https://papers.nips.cc/paper/5950-skip-thought-vectors.pdf)
paper. The following tasks are supported (refer to the paper for full details):

 * **SICK** semantic relatedness task.
 * **MSRP** (Microsoft Research Paraphrase Corpus) paraphrase detection task.
 * Binary classification tasks:
   * **MR** movie review sentiment task.
   * **CR** customer product review task.
   * **SUBJ** subjectivity/objectivity task.
   * **MPQA** opinion polarity task.
   * **TREC** question-type classification task.

### Preparation

You will need to clone or download the
[skip-thoughts GitHub repository](https://github.com/ryankiros/skip-thoughts) by
[ryankiros](https://github.com/ryankiros) (the first author of the Skip-Thoughts
paper):

```shell
# Folder to clone the repository to.
ST_KIROS_DIR="${HOME}/skip_thoughts/skipthoughts_kiros"

# Clone the repository.
git clone git@github.com:ryankiros/skip-thoughts.git "${ST_KIROS_DIR}/skipthoughts"

# Make the package importable.
export PYTHONPATH="${ST_KIROS_DIR}/:${PYTHONPATH}"
```

You will also need to download the data needed for each evaluation task. See the
instructions [here](https://github.com/ryankiros/skip-thoughts).

For example, the CR (customer review) dataset is found [here](http://nlp.stanford.edu/~sidaw/home/projects:nbsvm). For this task we want the
files `custrev.pos` and `custrev.neg`.

### Run the Evaluation Tasks

In the following example we will evaluate a unidirectional model ("uni-skip" in
the paper) on the CR task. To use a bidirectional model ("bi-skip" in the
paper),  simply pass the flags `--bi_vocab_file`, `--bi_embeddings_file` and
`--bi_checkpoint_path` instead. To use the "combine-skip" model described in the
paper you will need to pass both the unidirectional and bidirectional flags.

```shell
# Path to checkpoint file or a directory containing checkpoint files (the script
# will select the most recent).
CHECKPOINT_PATH="${HOME}/skip_thoughts/model/train"

# Vocabulary file generated by the vocabulary expansion script.
VOCAB_FILE="${HOME}/skip_thoughts/exp_vocab/vocab.txt"

# Embeddings file generated by the vocabulary expansion script.
EMBEDDINGS_FILE="${HOME}/skip_thoughts/exp_vocab/embeddings.npy"

# Directory containing files custrev.pos and custrev.neg.
EVAL_DATA_DIR="${HOME}/skip_thoughts/eval_data"

# Build the evaluation script.
cd tensorflow-models/skip_thoughts
bazel build -c opt //skip_thoughts:evaluate

# Run the evaluation script.
bazel-bin/skip_thoughts/evaluate \
  --eval_task=CR \
  --data_dir=${EVAL_DATA_DIR} \
  --uni_vocab_file=${VOCAB_FILE} \
  --uni_embeddings_file=${EMBEDDINGS_FILE} \
  --uni_checkpoint_path=${CHECKPOINT_PATH}
```

Output:

The output is a list of accuracies of 10 cross-validation classification models.
To get a single number, simply take the average:
