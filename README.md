## Character Classification Service

CCS takes an image and returns a small list of the most likely character tags found in the image.

### Training

The service has been trained on the 200,000 most recent uploads on Danbooru, filtered to posts that feature a single character. An Inception v4 deep network is trained on these images, with the character tags being the classes. Although the training process is being refined, the network sees an accuracy of about 70% and a recall rate of about 50%.

### Service

A simple Python Flask web server allows queries on the Inception network. 