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
"""Train the skip-thoughts model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import json

import configuration
import s2v_model

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string("input_file_pattern", None,
                       "File pattern of sharded TFRecord input files.")
tf.flags.DEFINE_string("checkpoint_dir", None,
                       "Directory containing model checkpoints.")
tf.flags.DEFINE_string("eval_dir", None, "Directory to write event logs to.")
tf.flags.DEFINE_string("master", "", "Eval master.")
tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
tf.flags.DEFINE_boolean("shuffle_input_data", False, "Whether to shuffle data")
tf.flags.DEFINE_integer("input_queue_capacity", 640000, "Input data queue capacity")
tf.flags.DEFINE_integer("num_input_reader_threads", 1, "Input data reader threads")
tf.flags.DEFINE_integer("eval_interval_secs", 600,
                        "Interval between evaluation runs.")
tf.flags.DEFINE_integer("num_eval_examples", 50000,
                        "Number of examples for evaluation.")
tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
tf.flags.DEFINE_integer("context_size", 1, "Prediction context size")
tf.flags.DEFINE_boolean("dropout", False, "Use dropout")
tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
tf.flags.DEFINE_string("model_config", None, "Model configuration json")

def main(unused_argv):
  if not FLAGS.input_file_pattern:
    raise ValueError("--input_file_pattern is required.")
  if not FLAGS.checkpoint_dir:
    raise ValueError("--checkpoint_dir is required.")
  if not FLAGS.eval_dir:
    raise ValueError("--eval_dir is required.")

  eval_dir = FLAGS.eval_dir
  if not tf.gfile.IsDirectory(eval_dir):
    tf.logging.info("Creating eval directory: %s", eval_dir)
    tf.gfile.MakeDirs(eval_dir)

  with open(FLAGS.model_config) as json_config_file:
    model_config = json.load(json_config_file)

  model_config = configuration.model_config(model_config, mode="eval")
  model = s2v_model.s2v(model_config, mode="eval")
  model.build()

  tf.summary.scalar("Loss", model.total_loss)
  summary_op = tf.summary.merge_all()

  config = tf.ConfigProto()
  config.gpu_options.allow_growth = True

  nsteps = int(FLAGS.num_eval_examples/FLAGS.batch_size)
  tf.contrib.slim.evaluation.evaluation_loop(
      master=FLAGS.master,
      checkpoint_dir=FLAGS.checkpoint_dir,
      logdir=FLAGS.eval_dir,
      num_evals=nsteps,
      eval_op=model.eval_op,
      summary_op=summary_op,
      eval_interval_secs=FLAGS.eval_interval_secs,
      session_config=config)

if __name__ == "__main__":
  tf.app.run()
