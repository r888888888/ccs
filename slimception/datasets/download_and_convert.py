import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import re
import os
import urllib.request
import tensorflow as tf
import math
import random
import sys
from datasets import dataset_utils
from itertools import islice
from pathlib import Path

_VALIDATION_PERCENTAGE = 0.9
_RANDOM_SEED = 0
_NUM_SHARDS = 5
_CSV_SOURCE_FILE = "posts.csv"
_NUM_CLASSES_FILE = "num_classes.txt"
_NUM_IMAGES_FILE = "num_images.txt"
_MIN_TERM_DF = 0.02
_MAX_TERM_DF = 0.3
_IGNORE_TAGS = set(["absurdres", "highres", "character_name", "character_request", "commentary", "commentary_request", "copyright_name", "official_art", "translation_request", "translated"])

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB JPEG data.                    
    self._decode_jpeg_data = tf.placeholder(dtype=tf.string)
    self._decode_jpeg = tf.image.decode_jpeg(self._decode_jpeg_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_jpeg(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_jpeg(self, sess, image_data):
    image = sess.run(
      self._decode_jpeg,
      feed_dict={self._decode_jpeg_data: image_data}
    )
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

def _get_dataset_filename(dataset_dir, split_name, shard_id):
  output_filename = 'chars_%s_%05d-of-%05d.tfrecord' % (
      split_name, shard_id, _NUM_SHARDS)
  return os.path.join(dataset_dir, output_filename)

def _tag_tokenizer(x):
  return x.split()

def _convert_dataset(split_name, hashes, class_names_to_ids, dataset_dir):
  assert split_name in ['train', 'validation']

  num_per_shard = int(math.ceil(len(hashes) / float(_NUM_SHARDS)))

  with tf.Graph().as_default():
    image_reader = ImageReader()

    with tf.Session('') as sess:

      for shard_id in range(_NUM_SHARDS):
        output_filename = _get_dataset_filename(
            dataset_dir, split_name, shard_id)

        with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
          start_ndx = shard_id * num_per_shard
          end_ndx = min((shard_id+1) * num_per_shard, len(hashes))
          for i in range(start_ndx, end_ndx):
            sys.stdout.write('\r>> Converting image %d/%d shard %d' % (
                i+1, len(hashes), shard_id))
            sys.stdout.flush()

            # Read the filename:
            image_data = tf.gfile.FastGFile(_image_path(dataset_dir, hashes[i]), 'rb').read()
            height, width = image_reader.read_image_dims(sess, image_data)

            num_classes = len(class_names_to_ids)
            class_names = _read_labels(dataset_dir, hashes[i])
            class_ids = np.zeros(num_classes, dtype=np.float32)
            for x in class_names:
              class_ids[class_names_to_ids[x]] = 1.0

            example = dataset_utils.image_to_tfexample(
                image_data, b'jpg', height, width, class_ids)
            tfrecord_writer.write(example.SerializeToString())

  sys.stdout.write('\n')
  sys.stdout.flush()

def _dataset_exists(dataset_dir):
  for split_name in ['train', 'validation']:
    for shard_id in range(_NUM_SHARDS):
      output_filename = _get_dataset_filename(
          dataset_dir, split_name, shard_id)
      if not tf.gfile.Exists(output_filename):
        return False
  return True

def _delete_all_labels(dataset_dir):
  for file in Path(os.path.normpath(os.path.join(dataset_dir, "..", "image_labels"))).iterdir():
    file.unlink()

def _delete_old_images(dataset_dir, hashes):
  for file in Path(os.path.normpath(os.path.join(dataset_dir, "..", "images"))).iterdir():
    h = re.search(r"[a-f0-9]{32}", str(file)).group(0)
    if h not in hashes:
      print("deleting", str(file))
      file.unlink()

def _download_images(dataset_dir):
  data = pd.read_csv(os.path.join(dataset_dir, _CSV_SOURCE_FILE))
  cv = CountVectorizer(min_df=_MIN_TERM_DF, max_df=_MAX_TERM_DF, tokenizer=_tag_tokenizer)
  cv.fit(data["tags"])
  tags = set(cv.vocabulary_.keys())
  hashes = set()

  with open(os.path.join(dataset_dir, _NUM_CLASSES_FILE), "w") as f:
    f.write(str(len(tags)))

  _delete_all_labels(dataset_dir)

  for index, row in data.iterrows():
    md5 = row["md5"]
    url = row["url"]
    ts = set(row["tags"].split(" "))
    ts = ts - _IGNORE_TAGS
    local_path = _image_path(dataset_dir, md5)
    label_path = _label_path(dataset_dir, md5)
    hashes.add(md5)
    if not os.path.isfile(local_path):
      print("downloading", url)
      urllib.request.urlretrieve(url, local_path)
    with open(label_path, "w") as f:
      f.write("\n".join(ts.intersection(tags)))

  _delete_old_images(dataset_dir, hashes)
  return (list(hashes), tags)

def _label_path(dataset_dir, hash):
  return os.path.normpath(os.path.join(dataset_dir, "..", "image_labels/{}.txt".format(hash)))

def _read_labels(dataset_dir, hash):
  with open(_label_path(dataset_dir, hash), "r") as f:
    return f.read().split()

def _image_path(dataset_dir, hash):
  return os.path.normpath(os.path.join(dataset_dir, "..", "images/{}.jpg".format(hash)))

def run(dataset_dir):
  if not tf.gfile.Exists(dataset_dir):
    tf.gfile.MakeDirs(dataset_dir)

  if _dataset_exists(dataset_dir):
    print('Dataset files already exist. Exiting without re-creating them.')
    return

  hashes, class_names = _download_images(dataset_dir)
  class_names_to_ids = dict(zip(class_names, range(len(class_names))))

  with open(os.path.join(dataset_dir, _NUM_IMAGES_FILE), "w") as f:
    f.write(str(len(hashes)))

  # Divide into train and test:
  random.seed(_RANDOM_SEED)
  random.shuffle(hashes)
  partition = int(len(hashes) * _VALIDATION_PERCENTAGE)
  training_hashes = hashes[partition:]
  validation_hashes = hashes[:partition]

  # First, convert the training and validation sets.
  _convert_dataset('train', training_hashes, class_names_to_ids,
                   dataset_dir)
  _convert_dataset('validation', validation_hashes, class_names_to_ids,
                   dataset_dir)

  # Finally, write the labels file:
  labels_to_class_names = dict(zip(range(len(class_names)), class_names))
  dataset_utils.write_label_file(labels_to_class_names, dataset_dir)

  print('\nFinished converting the chars dataset!')