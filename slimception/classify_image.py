#!/usr/bin/env python3

"""
based on http://warmspringwinds.github.io/tensorflow/tf-slim/2016/10/30/image-classification-and-segmentation-using-tensorflow-and-tf-slim/
"""

import numpy as np
import re
import os
import sys
import random
sys.path.append(os.path.normpath(os.path.join(__file__, "..", "..")))

import tensorflow as tf
from slimception.preprocessing import preprocessing_factory
from slimception.nets import nets_factory
from slimception.datasets import dataset_factory

slim = tf.contrib.slim

tf.app.flags.DEFINE_string('dataset_name', "tags", 'The name of the dataset.')
tf.app.flags.DEFINE_string('dataset_dir', '~/tf-data/dataset', 'path to dataset')
tf.app.flags.DEFINE_boolean('multilabel', True, 'Enable mutilabel mode')

FLAGS = tf.app.flags.FLAGS

def get_md5(string):
  return re.search(r"[a-f0-9]{32}", string).group(0)

def get_checkpoints_dir():
  return os.path.normpath(os.path.join(FLAGS.dataset_dir, "..", "models"))

def get_label_path(md5):
  return os.path.normpath(os.path.join(FLAGS.dataset_dir, "..", "image_labels", "{}.txt".format(md5)))

def get_random_image():
  files = os.listdir(os.path.join(FLAGS.dataset_dir, "..", "images"))
  # files = [x for x in files if "jpg" in x]
  return os.path.join(FLAGS.dataset_dir, "..", "images", random.choice(files))

def classify_image(path, labels, dataset, image_processing_fn, reuse):
  with open(path, "rb") as f:
    image = tf.image.decode_image(f.read(), channels=3)

  network_fn = nets_factory.get_network_fn(
    "inception_v4", 
    num_classes=dataset.num_classes,
    is_training=False,
    reuse=reuse
  )

  eval_image_size = network_fn.default_image_size
  processed_image = image_processing_fn(image, eval_image_size, eval_image_size)
  processed_image = tf.reshape(processed_image, (eval_image_size, eval_image_size, 3))
  processed_images = tf.expand_dims(processed_image, 0)
  logits, _ = network_fn(processed_images)
  if FLAGS.multilabel:
    probabilities = tf.nn.sigmoid(logits)
  else:
    probabilities = tf.nn.softmax(logits)
  checkpoint_path = tf.train.latest_checkpoint(get_checkpoints_dir())
  variables_to_restore = slim.get_variables_to_restore()
  init_fn = slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)

  with tf.Session() as sess:
    init_fn(sess)
    np_image, network_input, probabilities = sess.run([image, processed_image, probabilities])
    probabilities = probabilities[0, 0:]
    return sorted(zip(probabilities, labels), reverse=True)[0:3]

def main(_):
  with tf.Graph().as_default():
    dataset = dataset_factory.get_dataset(
      FLAGS.dataset_name, 
      "validation",
      FLAGS.dataset_dir
    )
    image_processing_fn = preprocessing_factory.get_preprocessing(
      "inception_v4",
      is_training=False,
      fast_mode=True
    )
    with open(os.path.join(FLAGS.dataset_dir, "labels.txt"), "r") as f:
      labels = [re.sub(r"^\d+:", "", x) for x in f.read().split()]

    reuse = False
    for i in range(1, 10):
      path = get_random_image()
      print(path)
      results = classify_image(path, labels, dataset, image_processing_fn, reuse)
      reuse = True
      md5 = get_md5(path)
      label_path = get_label_path(md5)
      with open(label_path, "r") as f:
        actual_label = " ".join(f.read().split())
      for score, label in results:
        print(label, "%.2f" % score)
      print("actual: {}".format(actual_label))


if __name__ == "__main__":
  tf.app.run()