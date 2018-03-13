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
"""Default configuration for model architecture and training."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
FLAGS = tf.flags.FLAGS

class _HParams(object):
  """Wrapper for configuration parameters."""
  pass

def vocab_config(vocab_config):
  config = _HParams()
  config.mode = vocab_config['mode']
  config.name = vocab_config['name']
  config.cap = vocab_config['cap']
  config.dim = vocab_config['dim']
  config.size = vocab_config['size']
  if config.mode == 'fixed':
    config.vocab_file = FLAGS.Glove_path
  else:
    config.vocab_file = FLAGS.results_path + vocab_config['vocab_file']
  config.embs_file = FLAGS.results_path + vocab_config['embs_file']
  return config

def model_config(mdl_config):

  config = _HParams()
  config.encoder = mdl_config['encoder']
  config.encoder_dim = mdl_config['encoder_dim']
  config.bidir = mdl_config['bidir']
  config.case_sensitive = mdl_config['case_sensitive']
  config.checkpoint_path = FLAGS.results_path + mdl_config['checkpoint_path']
  print(config.checkpoint_path)
  config.vocab_configs = []
  for vocab_configs in mdl_config['vocab_configs']:
    config.vocab_configs.append(vocab_config(vocab_configs))

  return config
