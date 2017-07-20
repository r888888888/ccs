import sys
import os
sys.path.append(os.path.normpath(os.path.join(__file__, "..", "..")))

import hashlib
import hmac
from flask import Flask
from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
import re
import numpy as np
import tensorflow as tf
import requests
import shutil
from slimception.datasets import characters
from slimception.preprocessing import preprocessing_factory
from slimception.nets import nets_factory
from slimception.datasets import dataset_factory

def initialize_graph(graph):
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
    return (dataset, image_processing_fn, session)

def allowed_file(filename):
  _, ext = os.path.splitext(filename.lower())
  return ext in set([".jpg", ".jpeg", ".png", ".gif"])

def query_inception(file, graph, labels, dataset, image_processing_fn, session):
  with graph.as_default():
    image = tf.image.decode_image(file.read(), channels=3)
    network_fn = nets_factory.get_network_fn(
      "inception_v4", 
      num_classes=dataset.num_classes,
      is_training=False,
      reuse=True
    )
    eval_image_size = network_fn.default_image_size
    processed_image = image_processing_fn(image, eval_image_size, eval_image_size)
    processed_image = tf.reshape(processed_image, (eval_image_size, eval_image_size, 3))
    processed_images = tf.expand_dims(processed_image, 0)
    logits, _ = network_fn(processed_images)
    probabilities = tf.nn.softmax(logits)
    probabilities = session.run([probabilities])
    probabilities = probabilities[0][0]
    guesses = sorted(zip(probabilities, labels), reverse=True)[0:5]
    return [(float(x[0]), x[1]) for x in guesses]

def initialize_globals():
  global _graph
  global _dataset
  global _image_processing_fn
  global _session
  global _labels

  _graph = tf.Graph()
  _dataset, _image_processing_fn, _session = initialize_graph(_graph)

  with open(os.path.join(os.environ.get("DATASET_DIR"), "labels.txt"), "r") as f:
    _labels = [re.sub(r"^\d+:", "", x) for x in f.read().split()]

def build_sig(msg):
  return hmac.new(environ.get("CCS_SECRET"), msg, hashlib.sha256).hexdigest()

def validate_params(url, ref, sig):
  msg = "{},{}".format(url, ref)
  msg_sig = build_sig(msg)
  return msg_sig == sig

def download_file(url, ref):
  r = requests.get(url, stream=True, headers={"Referer": ref})
  if r.status_code == requests.codes.ok:
    _, ext = os.path.splitext(url)
    file = tempfile.NamedTemporaryFile(suffix=ext, prefix="ccs-")
    for chunk in r.iter_content(1024):
      file.write(chunk)
    return file
  else:
    abort(400)

class ReverseProxied(object):
  '''Wrap the application in this middleware and configure the 
  front-end server to add these headers, to let you quietly bind 
  this to a URL other than / and to an HTTP scheme that is 
  different than what is used locally.

  In nginx:
  location /myprefix {
      proxy_pass http://192.168.0.1:5001;
      proxy_set_header Host $host;
      proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
      proxy_set_header X-Scheme $scheme;
      proxy_set_header X-Script-Name /myprefix;
      }

  :param app: the WSGI application
  '''
  def __init__(self, app):
    self.app = app

  def __call__(self, environ, start_response):
    script_name = environ.get('HTTP_X_SCRIPT_NAME', '')
    if script_name:
      environ['SCRIPT_NAME'] = script_name
      path_info = environ['PATH_INFO']
      if path_info.startswith(script_name):
        environ['PATH_INFO'] = path_info[len(script_name):]

    scheme = environ.get('HTTP_X_SCHEME', '')
    if scheme:
      environ['wsgi.url_scheme'] = scheme
    return self.app(environ, start_response)

load_dotenv("/etc/ccs/env")
initialize_globals()

app = Flask("ccs")
app.wsgi_app = ReverseProxied(app.wsgi_app)
app.config["UPLOAD_FOLDER"] = os.environ.get("FILE_UPLOAD_DIR")
app.config['MAX_CONTENT_LENGTH'] = 3 * 1024 * 1024

@app.route("/")
def index():
  return redirect(url_for("query"))

@app.route("/query.json", methods=["GET"])
def query_json():
  global _graph
  global _labels
  global _dataset
  global _image_processing_fn
  global _session

  url = request.args.get("url", "")
  ref = request.args.get("ref", "")
  sig = request.args.get("sig", "")

  if not validate_params(url, ref, sig):
    abort(401)

  if not allowed_file(url):
    abort(415)

  with download_file(url, ref) as f:
    answers = query_inception(f, _graph, _labels, _dataset, _image_processing_fn, _session)
    return json.dumps(answers)

@app.route("/query", methods=["GET", "POST"])
def query():
  global _graph
  global _labels
  global _dataset
  global _image_processing_fn
  global _session

  if request.method == "GET":
    return render_template("upload.html")

  # else method == POST
  if "file" not in request.files:
    flash("No file uploaded")
    return redirect(request.url)

  f = request.files["file"]
  if f.filename == '':
    flash("No file uploaded")
    return redirect(request.url)

  if not f or not allowed_file(f.filename)
    flash("Content type not supported")
    return redirect(request.url)

  answers = query_inception(f, _graph, _labels, _dataset, _image_processing_fn, _session)
  return render_template("results.html", answers=answers)

if __name__ == "__main__":
  app.run(debug=False, host="0.0.0.0")
