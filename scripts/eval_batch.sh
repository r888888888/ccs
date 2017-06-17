#!/bin/sh

find ~/tf-data-multi/images -name '00*' | head -n 10 | python3 slimception/classify_image.py
