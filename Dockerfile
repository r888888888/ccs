# you should run this with nvidia-docker

FROM danbooru/tensorflow:latest
MAINTAINER Albert Yi "r888888888@gmail.com"
RUN apt-get update -y
COPY . /app
WORKDIR /app
RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["web/ccs.py"]
