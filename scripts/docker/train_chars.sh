#!/bin/sh

nvidia-docker run -it -v /etc/ccs:/etc/ccs --entrypoint="/bin/bash" --cmd="scripts/train_chars.sh" ccs:latest
