# Use TensorFlow GPU image
FROM tensorflow/tensorflow:2.18.0-gpu

WORKDIR /app

# Copy and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade-strategy=only-if-needed -r requirements.txt

COPY aggregator.proto .

# Generate gRPC files
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. aggregator.proto

# Copy project files
COPY . .

# Generate gRPC files
RUN python -m grpc_tools.protoc -I. --python_out=. --grpc_python_out=. aggregator.proto


# Command to run the service (replace with actual script name)
CMD ["python", "image_input_service.py"]
