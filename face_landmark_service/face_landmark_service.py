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
import logging

mp_face_mesh = mp.solutions.face_mesh

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class FaceLandmarkDetectionService(aggregator_pb2_grpc.FaceLandmarkDetectionServicer):
    def __init__(self):
        self.redis_client = None
        logger.info("FaceLandmarkDetectionService initialized.")

    async def setup(self):
        """Setup the Redis client."""
        redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        logger.info(f"Connecting to Redis at {redis_url}")
        self.redis_client = await redis.asyncio.Redis.from_url(redis_url)
        logger.info("Redis client setup completed.")

    async def DetectLandmarks(self, request, context):
        """Detect face landmarks and store results."""
        logger.debug("Received DetectLandmarks request.")
        image_data = request.image_data

        image = cv2.imdecode(np.frombuffer(image_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            logger.warning("Failed to decode image.")
            return aggregator_pb2.LandmarkResponse(success=False)

        logger.debug("Image decoded successfully, converting to RGB.")
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image_hash = hashlib.md5(image_data).hexdigest()
        logger.debug(f"Generated image hash: {image_hash}")

        face_data = await self.process_landmarks(rgb_image, image.shape)

        if not face_data:
            logger.warning("No face landmarks detected.")
            return aggregator_pb2.LandmarkResponse(success=False)

        logger.info(f"Detected {len(face_data)} faces, storing results in Redis.")
        await self.redis_client.hset(image_hash, "landmarks", json.dumps(face_data))

        # Check for age/gender results and send face attributes if present
        if await self.redis_client.hexists(image_hash, "age_gender_results"):
            logger.info(f"Age/gender results found for image {image_hash}, sending face attributes.")
            asyncio.create_task(self.send_face_attributes(image_hash, image_data))

        return aggregator_pb2.LandmarkResponse(success=True)

    async def process_landmarks(self, rgb_image, image_shape):
        """Process and extract face landmarks using Mediapipe."""
        logger.debug("Processing face landmarks.")
        h, w, _ = image_shape
        face_data = []

        with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=5, refine_landmarks=True) as face_mesh:
            # Run face mesh processing in a separate thread
            results = await asyncio.to_thread(face_mesh.process, rgb_image)
            if not results.multi_face_landmarks:
                logger.info("No faces detected in image.")
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

        logger.debug(f"Processed {len(face_data)} faces.")
        return face_data

    async def send_face_attributes(self, image_hash, image_data):
        """Send face attributes to a separate service."""
        try:
            logger.debug(f"Sending face attributes for image {image_hash}.")
            async with grpc.aio.insecure_channel("data_storage_service:50053") as channel:
                stub = aggregator_pb2_grpc.DataStorageStub(channel)
                face_result = aggregator_pb2.FaceResult(
                    time=str(datetime.now()),
                    frame=image_data,
                    redis_key=image_hash,
                )
                await stub.SaveFaceAttributes(face_result)
                logger.info(f"Face attributes for image {image_hash} successfully sent.")
        except grpc.RpcError as e:
            logger.error(f"Error sending face attributes for image {image_hash}: {e}")

async def serve():
    """Start the gRPC server."""
    server = grpc.aio.server(options=[('grpc.max_metadata_size', 1024 * 1024)])
    service = FaceLandmarkDetectionService()
    await service.setup()

    aggregator_pb2_grpc.add_FaceLandmarkDetectionServicer_to_server(service, server)
    server.add_insecure_port("[::]:50051")
    logger.info("FaceLandmarkDetectionService running on port 50051...")

    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    logger.info("Starting FaceLandmarkDetectionService...")
    asyncio.run(serve())
