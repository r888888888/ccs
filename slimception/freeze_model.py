#!/usr/bin/env python3

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# based on https://blog.metaflow.fr/tensorflow-how-to-freeze-a-model-and-serve-it-with-a-python-api-d4f3596b3adc

import os, argparse
import tensorflow as tf
from tensorflow.python.framework import graph_util

def freeze_graph(model_dir):
  checkpoint = tf.train.get_checkpoint_state(model_dir)
  input_checkpoint = checkpoint.model_checkpoint_path
  absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
  output_graph = absolute_model_dir + "/frozen_model.pb"
  output_node_names = "AuxLogits,Logits"
  clear_devices = True
  saver = tf.train.import_meta_graph(input_checkpoint + ".meta", clear_devices=clear_devices)
  graph = tf.get_default_graph()
  input_graph_def = graph.as_graph_def()
  with tf.Session() as sess:
    saver.restore(sess, input_checkpoint)
    output_graph_def = graph_util.convert_variables_to_constants(sess, input_graph_def, output_node_names.split(","))
    with tf.gfile.GFile(output_graph, "wb") as f:
      f.write(output_graph_def.SerializeToString())
    print("%d ops in the final graph".format(len(output_graph_def.node)))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", type=str, help="Model directory to export")
  args = parser.parse_args()
  freeze_graph(args.model_dir)