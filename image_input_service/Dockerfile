FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed -r requirements.txt

COPY aggregator.proto .
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. aggregator.proto

COPY . .

CMD ["python", "image_input_service.py"]
