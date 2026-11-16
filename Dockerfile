FROM python:3.12-slim

WORKDIR /app

COPY requirements-service.txt .
RUN pip install --no-cache-dir -r requirements-service.txt

COPY train.py predict.py ./
COPY data/heart.csv data/heart.csv

# train the model inside the image so model.bin always matches the installed library versions
RUN python train.py

EXPOSE 9696

ENTRYPOINT ["gunicorn", "--bind=0.0.0.0:9696", "predict:app"]
