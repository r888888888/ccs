#!/bin/sh

DATA_HOME_DIR=${DATA_HOME_DIR:-/var/lib/ccs/data}
rm -rf $DATA_HOME_DIR/dataset/*.{txt,tfrecord}
rm -rf $DATA_HOME_DIR/models/*
