# you should run this with nvidia-docker

FROM tensorflow:latest-gpu-py3
MAINTAINER Albert Yi "r888888888@gmail.com"
RUN apt-get update -y
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
ENTRYPOINT ["python3"]
CMD ["web/test.py"]
