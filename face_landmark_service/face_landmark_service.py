import os
import json
import hashlib
from datetime import datetime

import grpc
import asyncio
import numpy as np
import cv2
import mediapipe as mp
import redis
import aggregator_pb2
import aggregator_pb2_grpc

mp_face_mesh = mp.solutions.face_mesh


class FaceLandmarkDetectionService(aggregator_pb2_grpc.FaceLandmarkDetectionServicer):
    def __init__(self):
        self.redis_client = None  # Initialize later in async context

    async def setup(self):
        """Initialize async Redis client."""
        self.redis_client = await redis.asyncio.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    async def DetectLandmarks(self, request, context):
        image_data = request.image_data

        # Decode image
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return aggregator_pb2.LandmarkResponse(success=False)

        # Convert image to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate a stable hash for the image
        image_hash = hashlib.md5(image_data).hexdigest()

        # Run face landmark detection asynchronously
        face_data = await self.process_landmarks(rgb_image, image.shape)

        if not face_data:
            return aggregator_pb2.LandmarkResponse(success=False)

        # Store face data in Redis
        await self.redis_client.hset(image_hash, "landmarks", json.dumps(face_data))

        # Check if age/gender data exists and send it to Data Storage Service
        if await self.redis_client.hexists(image_hash, "age_gender_results"):
            asyncio.create_task(self.send_face_attributes(image_hash, image_data))

        return aggregator_pb2.LandmarkResponse(success=True)

    async def process_landmarks(self, rgb_image, image_shape):
        """Run Mediapipe face landmark detection asynchronously."""
        h, w, _ = image_shape  # Get image dimensions
        face_data = []

        with mp_face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=5, refine_landmarks=True
        ) as face_mesh:
            results = await asyncio.to_thread(face_mesh.process, rgb_image)  # Run in separate thread

            if not results.multi_face_landmarks:
                return None

            for face_landmarks in results.multi_face_landmarks:
                points = [{"x": lm.x, "y": lm.y, "z": lm.z} for lm in face_landmarks.landmark]

                bbox = {
                    "x_min": int(min(lm.x for lm in face_landmarks.landmark) * w),
                    "y_min": int(min(lm.y for lm in face_landmarks.landmark) * h),
                    "x_max": int(max(lm.x for lm in face_landmarks.landmark) * w),
                    "y_max": int(max(lm.y for lm in face_landmarks.landmark) * h),
                }

                face_data.append({"landmarks": points, "bounding_box": bbox})

        return face_data

    async def send_face_attributes(self, image_hash, image_data):
        """Send detected face attributes to Data Storage Service asynchronously."""
        try:
            async with grpc.aio.insecure_channel("data_storage_service:50053") as channel:
                stub = aggregator_pb2_grpc.DataStorageStub(channel)
                face_result = aggregator_pb2.FaceResult(
                    time=str(datetime.now()),
                    frame=image_data,
                    redis_key=image_hash,
                )
                await stub.SaveFaceAttributes(face_result)  # Non-blocking gRPC request
                print("Face attributes sent successfully.")
        except grpc.RpcError as e:
            print(f"Error sending face attributes: {e}")


async def serve():
    server = grpc.aio.server(options=[('grpc.max_metadata_size', 1024 * 1024)])
    service = FaceLandmarkDetectionService()
    await service.setup()  # Initialize Redis client

    aggregator_pb2_grpc.add_FaceLandmarkDetectionServicer_to_server(service, server)
    server.add_insecure_port("[::]:50051")
    print("FaceLandmarkDetectionService running on port 50051...")

    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
