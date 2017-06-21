#!/usr/bin/env python3

# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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
r"""Downloads and converts a particular dataset.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from datasets import download_and_convert

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string(
  'dataset_dir',
  None,
  'The directory where the output TFRecords and temporary files are saved.'
)

tf.app.flags.DEFINE_string(
  'source_csv',
  'posts.csv',
  'Source CSV file with all the data'
)

tf.app.flags.DEFINE_string(
  'num_classes_file',
  None,
  "Text file indicating how many classes the data set has"
)

tf.app.flags.DEFINE_string(
  'num_images_file',
  None,
  "Text file indicating how many images the data set has"
)

tf.app.flags.DEFINE_string(
  'daataset_name',
  None,
  "Name of the data set"
)

def main(_):
  if not FLAGS.dataset_dir:
    raise ValueError('You must supply the dataset directory with --dataset_dir')

  processor = download_and_convert.DownloaderAndConverter(
    num_classes_file=FLAGS.num_classes_file,
    num_images_file=FLAGS.num_images_file,
    dataset_name=FLAGS.dataset_name,
    dataset_dir='~/tf-data-multi',
    source_csv=FLAGS.source_csv
  )
  processor.run()

if __name__ == '__main__':
  tf.app.run()
