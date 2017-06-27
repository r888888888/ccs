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
import time
import http
import multiprocessing
from functools import partial

def _tag_tokenizer(x):
  return x.split()

class ImageReader(object):
  """Helper class that provides TensorFlow image coding utilities."""

  def __init__(self):
    # Initializes function that decodes RGB image data.                    
    self._decode_image_data = tf.placeholder(dtype=tf.string)
    self._decode_image = tf.image.decode_image(self._decode_image_data, channels=3)

  def read_image_dims(self, sess, image_data):
    image = self.decode_image(sess, image_data)
    return image.shape[0], image.shape[1]

  def decode_image(self, sess, image_data):
    image = sess.run(
      self._decode_image,
      feed_dict={self._decode_image_data: image_data}
    )
    assert len(image.shape) == 3
    assert image.shape[2] == 3
    return image

class DownloaderAndConverter():
  def __init__(self, **kwargs):
    self._num_classes_file = kwargs.get('num_classes_file', 'num_classes.txt')
    self._num_images_file = kwargs.get('num_images_file', 'num_images.txt')
    self._num_shards = kwargs.get('num_shards', 5)
    self._dataset_name = kwargs.get('dataset_name')
    self._dataset_dir = kwargs.get('dataset_dir', '~/tf-data')
    self._source_csv = kwargs.get('source_csv', 'posts.csv')
    self._random_seed = kwargs.get('random_seed', 42)
    self._min_term_df = kwargs.get('min_term_df', 0.02)
    self._max_term_df = kwargs.get('max_term_df', 0.2)
    self._ignore_tags = kwargs.get('ignore_tags', set())
    self._validation_percentage = kwargs.get('validation_percentage', 0.9)
    self._multilabel = kwargs.get('multilabel', False)

    if self._multilabel:
      sys.stdout.write(">> Enabling multi-label mode\n")
      sys.stdout.flush()

  def _get_dataset_filename(self, split_name, shard_id):
    output_filename = '%s_%s_%05d-of-%05d.tfrecord' % (
      self._dataset_name,
      split_name, 
      shard_id, 
      self._num_shards
    )
    return os.path.join(self._dataset_dir, output_filename)

  def _convert_dataset(self, split_name, hashes, class_names_to_ids):
    assert split_name in ['train', 'validation']

    num_per_shard = int(math.ceil(len(hashes) / float(self._num_shards)))

    with tf.Graph().as_default():
      image_reader = ImageReader()

      with tf.Session('') as sess:

        for shard_id in range(self._num_shards):
          output_filename = self._get_dataset_filename(split_name, shard_id)

          with tf.python_io.TFRecordWriter(output_filename) as tfrecord_writer:
            start_ndx = shard_id * num_per_shard
            end_ndx = min((shard_id+1) * num_per_shard, len(hashes))
            for i in range(start_ndx, end_ndx):
              sys.stdout.write(
                '\r>> Converting image %d/%d shard %d [%s]\n' % (
                  i+1, 
                  len(hashes), 
                  shard_id,
                  hashes[i]
                )
              )
              sys.stdout.flush()

              # Read the filename:
              image_path = self._image_path(hashes[i])
              image_data = tf.gfile.FastGFile(image_path, 'rb').read()
              try:
                height, width = image_reader.read_image_dims(sess, image_data)

                num_classes = len(class_names_to_ids)
                if self._multilabel:
                  class_names = self._read_labels(hashes[i])
                  class_ids = np.zeros(num_classes, dtype=np.int64)
                  for x in class_names:
                    class_ids[class_names_to_ids[x]] = 1
                else:
                  class_name = self._read_labels(hashes[i])
                  class_ids = class_names_to_ids[class_name]

                ext = image_path.split(".")[-1]

                example = dataset_utils.image_to_tfexample(
                  image_data, 
                  ext.encode(), 
                  height, 
                  width, 
                  class_ids
                )
                tfrecord_writer.write(example.SerializeToString())
              except tf.errors.InvalidArgumentError:
                print("error reading image")

    sys.stdout.write('\n')
    sys.stdout.flush()

  def _dataset_exists(self):
    for split_name in ['train', 'validation']:
      for shard_id in range(self._num_shards):
        output_filename = self._get_dataset_filename(split_name, shard_id)
        if not tf.gfile.Exists(output_filename):
          return False
    return True

  def _delete_all_labels(self):
    for file in Path(os.path.normpath(os.path.join(self._dataset_dir, "..", "image_labels"))).iterdir():
      file.unlink()

  def _delete_old_images(self, hashes):
    for file in Path(os.path.normpath(os.path.join(self._dataset_dir, "..", "images"))).iterdir():
      h = re.search(r"[a-f0-9]{32}", str(file)).group(0)
      if h not in hashes:
        print("deleting", str(file))
        file.unlink()

  def _download_image(self, args, tags=None, hashes=None):
    row = args[1]
    md5 = row["md5"]
    url = row["url"]
    ts = set(row["tags"].split(" "))
    ts = ts - self._ignore_tags
    if len(ts) > 0:
      ext = url.split(".")[-1]
      local_path = self._image_path(md5, ext)
      label_path = self._label_path(md5)
      hashes.add(md5)
    if not os.path.isfile(local_path):
      print("downloading", url)
      while True:
        try:
          urllib.request.urlretrieve(url, local_path)
        except http.client.RemoteDisconnected:
          time.sleep(5)
          print("  remote disconnected")
          continue
        break
    with open(label_path, "w") as f:
      f.write("\n".join(ts.intersection(tags)))

  def _download_images(self):
    hashes = set()
    data = pd.read_csv(os.path.join(self._dataset_dir, self._source_csv))
    data.dropna(inplace=True, how='any')
    cv = CountVectorizer(min_df=self._min_term_df, max_df=self._max_term_df, tokenizer=_tag_tokenizer)
    cv.fit(data["tags"])
    tags = set(cv.vocabulary_.keys())

    with open(os.path.join(self._dataset_dir, self._num_classes_file), "w") as f:
      f.write(str(len(tags)))

    self._delete_all_labels()
    pool = multiprocessing.Pool(processes=8)
    download_image_wrapper = partial(self._download_image, tags=tags, hashes=hashes)
    pool.imap_unordered(download_image_wrapper, data.iterrows())
    pool.close()
    pool.join()

    #self._delete_old_images(hashes)
    return (list(hashes), tags)

  def _label_path(self, hash):
    return os.path.normpath(os.path.join(self._dataset_dir, "..", "image_labels/{}.txt".format(hash)))

  def _read_labels(self, hash):
    with open(self._label_path(hash), "r") as f:
      if self._multilabel:
        return f.read().split()
      else:
        return f.read()

  def _image_path(self, hash, ext=None):
    if ext is None:
      for x in ["jpg", "png"]:
        if os.path.isfile(self._image_path(hash, x)):
          ext = x
    return os.path.normpath(os.path.join(self._dataset_dir, "..", "images/{}.{}".format(hash, ext)))

  def run(self):
    if not tf.gfile.Exists(self._dataset_dir):
      tf.gfile.MakeDirs(self._dataset_dir)

    if self._dataset_exists():
      print('Dataset files already exist. Exiting without re-creating them.')
      return

    hashes, class_names = self._download_images()
    class_names_to_ids = dict(zip(class_names, range(len(class_names))))

    with open(os.path.join(self._dataset_dir, self._num_images_file), "w") as f:
      f.write(str(len(hashes)))

    # Divide into train and test:
    random.seed(self._random_seed)
    random.shuffle(hashes)
    partition = int(len(hashes) * self._validation_percentage)
    training_hashes = hashes[partition:]
    validation_hashes = hashes[:partition]

    # First, convert the training and validation sets.
    self._convert_dataset(
      'train', 
      training_hashes, 
      class_names_to_ids
    )
    self._convert_dataset(
      'validation', 
      validation_hashes, 
      class_names_to_ids
    )

    # Finally, write the labels file:
    labels_to_class_names = dict(zip(range(len(class_names)), class_names))
    dataset_utils.write_label_file(labels_to_class_names, self._dataset_dir)

    print('\nFinished converting the dataset!')