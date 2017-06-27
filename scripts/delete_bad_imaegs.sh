#!/bin/sh

find images -name '*.jpg' | jpeginfo -c -d -f-
find images -name '*.png' | pngcheck -q > pngerrors.txt
