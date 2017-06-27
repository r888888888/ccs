#!/bin/sh

DATA_HOME_DIR=${DATA_HOME_DIR:-~/tf-data-multi}
find $DATA_HOME_DIR/images -name '*.jpg' | jpeginfo -c -d -f-
find $DATA_HOME_DIR/images -name '*.png' | pngcheck -q > pngerrors.txt
