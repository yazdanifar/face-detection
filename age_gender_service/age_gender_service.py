import os
import json
import hashlib
from datetime import datetime

import grpc
import asyncio
import aggregator_pb2
import aggregator_pb2_grpc
from deepface import DeepFace
import cv2
import numpy as np
import redis


class AgeGenderEstimationService(aggregator_pb2_grpc.AgeGenderEstimationServicer):

    def __init__(self):
        self.redis_client = redis.asyncio.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    async def EstimateAgeGender(self, request, context):
        image_data = request.image_data

        # Offload image decoding to a separate thread to avoid blocking the event loop
        image = await asyncio.to_thread(self.decode_image, image_data)
        if image is None:
            return aggregator_pb2.AgeGenderResponse(success=False)

        image_hash = hashlib.md5(image_data).hexdigest()

        # Offload DeepFace analysis to a separate thread to avoid blocking the event loop
        analysis = await asyncio.to_thread(self.analyze_age_gender, image)
        if not analysis:
            return aggregator_pb2.AgeGenderResponse(success=False)

        json_results = []

        for result in analysis:
            gender = max(result['gender'], key=result['gender'].get)
            age = result['age']
            region = result['region']
            json_results.append({
                "age": age,
                "gender": gender,
                "bounding_box": {
                    "x_min": region['x'],
                    "y_min": region['y'],
                    "x_max": region['x'] + region['w'],
                    "y_max": region['y'] + region['h'],
                }
            })

        # Store results in Redis asynchronously
        await self.redis_client.hset(image_hash, "age_gender_results", json.dumps(json_results))

        # Check for existing landmarks in Redis and send data if present
        if await self.redis_client.hexists(image_hash, "landmarks"):
            async with grpc.aio.insecure_channel('data_storage_service:50053') as channel:
                stub = aggregator_pb2_grpc.DataStorageStub(channel)
                face_result = aggregator_pb2.FaceResult(
                    time=str(datetime.now()),
                    frame=image_data,
                    redis_key=image_hash
                )
                # Send the request asynchronously and don't wait for the response
                await stub.SaveFaceAttributes(face_result)
                print("Request sent, not waiting for response!")

        return aggregator_pb2.AgeGenderResponse(success=True)

    def decode_image(self, image_data):
        """Decodes the image, used in a separate thread to avoid blocking the event loop."""
        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        return image

    def analyze_age_gender(self, image):
        """Analyze age and gender, used in a separate thread to avoid blocking the event loop."""
        try:
            # Analyze age and gender using DeepFace
            analysis = DeepFace.analyze(image, actions=['gender', 'age'], enforce_detection=True)
            return analysis
        except Exception as e:
            print(f"Error in DeepFace analysis: {e}")
            return None


async def serve():
    server = grpc.aio.server(options=[('grpc.max_metadata_size', 1024 * 1024)])
    aggregator_pb2_grpc.add_AgeGenderEstimationServicer_to_server(AgeGenderEstimationService(), server)
    server.add_insecure_port('[::]:50052')
    print("AgeGenderEstimationService running on port 50052...")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
