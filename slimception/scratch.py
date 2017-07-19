import sys
import os
sys.path.append("/var/lib/ccs/app")
from dotenv import load_dotenv
import json
import re
import numpy as np
import tensorflow as tf
from slimception.datasets import characters
from slimception.preprocessing import preprocessing_factory
from slimception.nets import nets_factory
from slimception.datasets import dataset_factory

load_dotenv("/etc/ccs/env")

graph = tf.Graph()

with graph.as_default():
  dataset = dataset_factory.get_dataset(
    "characters", 
    "validation",
    os.environ.get("DATASET_DIR")
  )
  image_processing_fn = preprocessing_factory.get_preprocessing(
    "inception_v4",
    is_training=False
  )
  checkpoint_path = tf.train.latest_checkpoint(os.environ.get("CHECKPOINTS_DIR"))
  network_fn = nets_factory.get_network_fn(
    "inception_v4", 
    num_classes=dataset.num_classes,
    is_training=False,
    reuse=None
  )
  eval_image_size = network_fn.default_image_size
  placeholder = tf.expand_dims(tf.placeholder(tf.float32, shape=(eval_image_size, eval_image_size, 3)), 0)
  network_fn(placeholder)
  variables_to_restore = tf.contrib.slim.get_variables_to_restore()
  init_fn = tf.contrib.slim.assign_from_checkpoint_fn(checkpoint_path, variables_to_restore)
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
  tf_sess_config = tf.ConfigProto(gpu_options=gpu_options)
  session = tf.Session(graph=graph, config=tf_sess_config)
  init_fn(session)

with graph.as_default():
  with tf.gfile.GFile("/tmp/200px-Th12Reimu.png", "rb") as file:
    image = tf.image.decode_image(file.read(), channels=3)
  print("loaded file")
  processed_image = image_processing_fn(image, eval_image_size, eval_image_size)
  processed_image = tf.reshape(processed_image, (eval_image_size, eval_image_size, 3))
  processed_images = tf.expand_dims(processed_image, 0)
  network_fn = nets_factory.get_network_fn(
    "inception_v4", 
    num_classes=dataset.num_classes,
    is_training=False,
    reuse=True
  )
  print("generating logits")
  logits, _ = network_fn(processed_images)
  probabilities = tf.nn.softmax(logits)
  print("running inference")
  probabilities = session.run([probabilities])
  print(probabilities)
