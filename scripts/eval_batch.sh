#!/bin/sh

DATA_HOME_DIR=${DATA_HOME_DIR:-~/tf-data-multi}

find ${DATA_HOME_DIR}/images | shuf | head -n 5 | python3 slimception/classify_image.py
