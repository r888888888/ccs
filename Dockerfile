# you should run this with nvidia-docker

FROM tensorflow/tensorflow:latest-gpu-py3
MAINTAINER Albert Yi "r888888888@gmail.com"
RUN apt-get update -y
RUN apt-get install -y libmagic-dev
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["gunicorn -w 3 -b 127.0.0.1:5000 web.ccs:app"]
