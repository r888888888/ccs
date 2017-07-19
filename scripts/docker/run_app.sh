#!/bin/sh

nvidia-docker run -d -p 5000:5000 -v /etc/ccs:/etc/ccs -v /var/lib/ccs/data:/var/lib/ccs/data ccs:latest
