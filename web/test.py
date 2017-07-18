from flask import Flask
from flask import request, render_template, redirect, url_for
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json
import re
import numpy as np
import tensorflow as tf
# from slimception.datasets import characters
# from slimception.preprocessing import preprocessing_factory
# from slimception.nets import nets_factory
# from slimception.datasets import dataset_factory

app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Flask Dockerized'

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

