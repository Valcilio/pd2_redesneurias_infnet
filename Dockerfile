FROM python:3.13

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --upgrade pip

RUN pip install -r requirements.txt

EXPOSE 5000

COPY ./api.py /app/api.py

COPY ./model_pipeline.py /app/model_pipeline.py

COPY ./model.keras /app/model.keras

ENTRYPOINT ["python"]

CMD ["api.py"]