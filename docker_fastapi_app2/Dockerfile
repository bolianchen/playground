FROM python:3.10

WORKDIR /home/app

COPY ./requirements.txt ./requirements.txt

RUN pip install -r https://raw.githubusercontent.com/ultralytics/yolov5/master/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

EXPOSE 8000

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

COPY . .

CMD ["uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8000"]
