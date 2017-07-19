#!/bin/sh

nvidia-docker run -d -p 5000:5000 -v /etc/ccs:/etc/ccs ccs:latest
