## Character Classification Service

CCS takes an image and returns a small list of the most likely character tags found in the image.

### Training

The service has been trained on recent uploads from Danbooru. An Inception v4 deep network is trained on these images, with the character tags being the classes.

### Service

A simple Python Flask web server allows queries on the network. 

### Setup

This service assumes your server has an Nvidia GPU to accelerate training and inference. An example script in scripts/install.sh is provided to setup Docker, and install the relevant Nvidia drivers to enable the GPU accelerated version of Tensorflow.

The provided scripts assume the app will be install on /var/lib/ccs/app and the data will be stored in /var/lib/ccs/data. An example Ruby script is provided at scripts/export_from_danbooru.rb to generate a CSV file from the Danbooru database, picking only images with a single character tag to make training more accurate. You should put the CSV file in /var/lib/ccs/data/dataset.

You can run scripts/docker/build_app.sh to build a local copy of the Docker container.

You can then run scripts/docker/train_chars.sh to download the images and train your network.

The current web service is under work; it will eventually be replaced with a Tensorflow Serving server to speed up inference.
