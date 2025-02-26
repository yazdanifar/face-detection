# Use TensorFlow GPU image
FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed -r requirements.txt

COPY ../aggregator.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. aggregator.proto

COPY . .

EXPOSE 50052

CMD ["python", "age_gender_service.py"]
