# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""

The evaluation tasks have different running times. SICK may take 5-10 minutes.
MSRP, TREC and CR may take 20-60 minutes. SUBJ, MPQA and MR may take 2+ hours.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import eval_classification
import eval_msrp
import eval_sick
import eval_trec

import tensorflow as tf
import numpy as np

import configuration
import encoder_manager

import json

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("eval_task", "MSRP",
                       "Name of the evaluation task to run. Available tasks: "
                       "MR, CR, SUBJ, MPQA, SICK, MSRP, TREC.")
tf.flags.DEFINE_string("data_dir", None, "Directory containing training data.")
tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
tf.flags.DEFINE_boolean("use_norm", False,
                        "Normalize sentence embeddings during evaluation")
tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
tf.flags.DEFINE_string("model_config", None, "Model configuration json")
tf.flags.DEFINE_string("results_path", None, "Model results path")
tf.flags.DEFINE_string("Glove_path", None, "Path to Glove dictionary")

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  if not FLAGS.data_dir:
    raise ValueError("--data_dir is required.")
  if not FLAGS.model_config:
    raise ValueError("--model_config is required.")

  encoder = encoder_manager.EncoderManager()

  with open(FLAGS.model_config) as json_config_file:
    model_config = json.load(json_config_file)

  if type(model_config) is dict:
    model_config = [model_config]

  for mdl_cfg in model_config:
    model_config = configuration.model_config(mdl_cfg, mode="encode")
    encoder.load_model(model_config)

  if FLAGS.eval_task in ["MR", "CR", "SUBJ", "MPQA"]:
    results = eval_classification.eval_nested_kfold(
        encoder, FLAGS.eval_task, FLAGS.data_dir, use_nb=False)
    scores = results[0]
    print('Mean score', np.mean(scores))
  elif FLAGS.eval_task == "SICK":
    results = eval_sick.evaluate(encoder, evaltest=True, loc=FLAGS.data_dir)
  elif FLAGS.eval_task == "MSRP":
    results = eval_msrp.evaluate(
        encoder, evalcv=True, evaltest=True, use_feats=False, loc=FLAGS.data_dir)
  elif FLAGS.eval_task == "TREC":
    eval_trec.evaluate(encoder, evalcv=True, evaltest=True, loc=FLAGS.data_dir)
  else:
    raise ValueError("Unrecognized eval_task: %s" % FLAGS.eval_task)

  encoder.close()

if __name__ == "__main__":
  tf.app.run()
