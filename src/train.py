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
                       "File pattern of sharded TFRecord files containing")
tf.flags.DEFINE_string("train_dir", None,
                       "Directory for saving and loading checkpoints.")
tf.flags.DEFINE_integer("batch_size", 400, "Batch size")
tf.flags.DEFINE_float("uniform_init_scale", 0.1, "Random init scale")
tf.flags.DEFINE_boolean("shuffle_input_data", False, "Whether to shuffle data")
tf.flags.DEFINE_integer("input_queue_capacity", 640000, "Input data queue capacity")
tf.flags.DEFINE_integer("num_input_reader_threads", 1, "Input data reader threads")
tf.flags.DEFINE_float("learning_rate", 0.0005, "Learning rate")
tf.flags.DEFINE_integer("learning_rate_decay_steps", 400000, "Learning rate decay steps")
tf.flags.DEFINE_float("clip_gradient_norm", 5.0, "Gradient clipping norm")
tf.flags.DEFINE_integer("save_model_secs", 600, "Checkpointing frequency")
tf.flags.DEFINE_integer("save_summaries_secs", 600, "Summary frequency")
tf.flags.DEFINE_integer("nepochs", 1, "Number of epochs")
tf.flags.DEFINE_integer("num_train_inst", 45786400, "Number of training instances")
tf.flags.DEFINE_integer("sequence_length", 30, "Max sentence length considered")
tf.flags.DEFINE_integer("context_size", 1, "Prediction context size")
tf.flags.DEFINE_boolean("dropout", False, "Use dropout")
tf.flags.DEFINE_float("dropout_rate", 0.3, "Dropout rate")
tf.flags.DEFINE_string("model_config", None, "Model configuration json")
tf.flags.DEFINE_integer("max_ckpts", 5, "Max number of ckpts to keep")
tf.flags.DEFINE_string("Glove_path", None, "Path to Glove dictionary")

tf.logging.set_verbosity(tf.logging.INFO)

def main(unused_argv):
  if not FLAGS.input_file_pattern:
    raise ValueError("--input_file_pattern is required.")
  if not FLAGS.train_dir:
    raise ValueError("--train_dir is required.")

  with open(FLAGS.model_config) as json_config_file:
    model_config = json.load(json_config_file)

  model_config = configuration.model_config(model_config, mode="train")
  tf.logging.info("Building training graph.")
  g = tf.Graph()
  with g.as_default():
    model = s2v_model.s2v(model_config, mode="train")
    model.build()

    optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)

    train_tensor = tf.contrib.slim.learning.create_train_op(
        total_loss=model.total_loss,
        optimizer=optimizer,
        clip_gradient_norm=FLAGS.clip_gradient_norm)
        #global_step=model.global_step,

    if FLAGS.max_ckpts != 5:
      saver = tf.train.Saver(max_to_keep=FLAGS.max_ckpts)
    else:
      saver = tf.train.Saver()

  load_words = model.init
  if load_words:
    def InitAssignFn(sess):
      sess.run(load_words[0], {load_words[1]: load_words[2]})

  nsteps = int(FLAGS.nepochs * (FLAGS.num_train_inst / FLAGS.batch_size))
  tf.contrib.slim.learning.train(
      train_op=train_tensor,
      logdir=FLAGS.train_dir,
      graph=g,
      number_of_steps=nsteps,
      save_summaries_secs=FLAGS.save_summaries_secs,
      saver=saver,
      save_interval_secs=FLAGS.save_model_secs, 
      init_fn=InitAssignFn if load_words else None
  )

if __name__ == "__main__":
  tf.app.run()
