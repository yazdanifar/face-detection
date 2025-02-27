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
import logging

# Set up logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class AgeGenderEstimationService(aggregator_pb2_grpc.AgeGenderEstimationServicer):

    def __init__(self):
        self.redis_client = redis.asyncio.Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        logger.info("Initialized AgeGenderEstimationService")

    async def EstimateAgeGender(self, request, context):
        image_data = request.image_data
        logger.info("Received request to estimate age and gender for image")

        # Offload image decoding to a separate thread to avoid blocking the event loop
        image = await asyncio.to_thread(self.decode_image, image_data)
        if image is None:
            logger.error("Failed to decode image")
            return aggregator_pb2.AgeGenderResponse(success=False)

        image_hash = hashlib.md5(image_data).hexdigest()
        logger.info(f"Generated image hash: {image_hash}")

        # Offload DeepFace analysis to a separate thread to avoid blocking the event loop
        analysis = await asyncio.to_thread(self.analyze_age_gender, image)
        if not analysis:
            logger.error("Failed to analyze age and gender using DeepFace")
            return aggregator_pb2.AgeGenderResponse(success=False)

        json_results = []
        logger.info(f"Analysis successful, processing {len(analysis)} results")

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
        logger.info(f"Processed {len(json_results)} face attributes")

        # Store results in Redis asynchronously
        try:
            await self.redis_client.hset(image_hash, "age_gender_results", json.dumps(json_results))
            logger.info(f"Stored age and gender results in Redis with key {image_hash}")
        except Exception as e:
            logger.error(f"Error storing data in Redis: {e}")
            return aggregator_pb2.AgeGenderResponse(success=False)

        # Check for existing landmarks in Redis and send data if present
        if await self.redis_client.hexists(image_hash, "landmarks"):
            try:
                async with grpc.aio.insecure_channel('data_storage_service:50053') as channel:
                    stub = aggregator_pb2_grpc.DataStorageStub(channel)
                    face_result = aggregator_pb2.FaceResult(
                        time=str(datetime.now()),
                        frame=image_data,
                        redis_key=image_hash
                    )
                    # Send the request asynchronously and don't wait for the response
                    await stub.SaveFaceAttributes(face_result)
                    logger.info(f"Sent face attributes to data storage service for image hash {image_hash}")
            except grpc.aio.AioRpcError as e:
                logger.error(f"Failed to send face attributes to data storage service: {e}")
                raise

        return aggregator_pb2.AgeGenderResponse(success=True)

    def decode_image(self, image_data):
        """Decodes the image, used in a separate thread to avoid blocking the event loop."""
        try:
            image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
            logger.info("Image decoded successfully")
            return image
        except Exception as e:
            logger.error(f"Error decoding image: {e}")
            raise

    def analyze_age_gender(self, image):
        """Analyze age and gender, used in a separate thread to avoid blocking the event loop."""
        try:
            # Analyze age and gender using DeepFace
            logger.info("Starting DeepFace analysis for age and gender")
            analysis = DeepFace.analyze(image, actions=['gender', 'age'], enforce_detection=True)
            logger.info("DeepFace analysis completed successfully")
            return analysis
        except Exception as e:
            logger.error(f"Error in DeepFace analysis: {e}")
            raise


async def serve():
    server = grpc.aio.server(options=[('grpc.max_metadata_size', 1024 * 1024)])
    aggregator_pb2_grpc.add_AgeGenderEstimationServicer_to_server(AgeGenderEstimationService(), server)
    server.add_insecure_port('[::]:50052')
    logger.info("AgeGenderEstimationService running on port 50052...")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logger.info("Starting the AgeGenderEstimationService")
    asyncio.run(serve())
