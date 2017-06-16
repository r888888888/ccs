#!/bin/sh

export TF_HOME_DIR=~/tf-data
rm -rf $TF_HOME_DIR/dataset/*.{txt,tfrecord}
rm -rf $TF_HOME_DIR/models/*
