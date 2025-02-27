import aiofiles
import os
import json
import grpc
import redis.asyncio as redis
import numpy as np
from scipy.optimize import linear_sum_assignment
import asyncio
import io
from PIL import Image, ImageDraw, ImageFont
import logging

import aggregator_pb2
import aggregator_pb2_grpc

# Set up logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
    logger.debug("Calculating IoU for boxes: %s and %s", boxA, boxB)
    xA = max(boxA["x_min"], boxB["x_min"])
    yA = max(boxA["y_min"], boxB["y_min"])
    xB = min(boxA["x_max"], boxB["x_max"])
    yB = min(boxA["y_max"], boxB["y_max"])

    # Compute the intersection area
    inter_area = max(0, xB - xA) * max(0, yB - yA)

    # Compute the area of both bounding boxes
    boxA_area = (boxA["x_max"] - boxA["x_min"]) * (boxA["y_max"] - boxA["y_min"])
    boxB_area = (boxB["x_max"] - boxB["x_min"]) * (boxB["y_max"] - boxB["y_min"])

    # Compute IoU
    iou_value = inter_area / float(boxA_area + boxB_area - inter_area) if (
                                                                                      boxA_area + boxB_area - inter_area) > 0 else 0
    logger.debug("Calculated IoU: %f", iou_value)
    return iou_value


async def match_faces_with_hungarian(landmarks, age_gender_results):
    """
    Match faces using the Hungarian algorithm based on IoU.
    """
    logger.info("Starting face matching using Hungarian algorithm.")
    num_landmarks = len(landmarks)
    num_age_gender = len(age_gender_results)
    cost_matrix = np.zeros((num_landmarks, num_age_gender))

    for i, landmark in enumerate(landmarks):
        for j, age_gender in enumerate(age_gender_results):
            cost_matrix[i, j] = 1 - iou(landmark["bounding_box"], age_gender["bounding_box"])  # Use 1 - IoU as cost

    # Apply the Hungarian algorithm
    landmark_indices, age_gender_indices = linear_sum_assignment(cost_matrix)

    matched_faces = []
    for i, j in zip(landmark_indices, age_gender_indices):
        if cost_matrix[i, j] < 0.5:  # Threshold for matching (IoU > 0.5)
            matched_faces.append({
                "landmarks": landmarks[i]["landmarks"],
                "bounding_box": landmarks[i]["bounding_box"],
                "age": age_gender_results[j]["age"],
                "gender": age_gender_results[j]["gender"]
            })
        else:
            matched_faces.append({
                "landmarks": landmarks[i]["landmarks"],
                "bounding_box": landmarks[i]["bounding_box"],
                "age": None,
                "gender": None
            })

    logger.info("Face matching completed, found %d matches.", len(matched_faces))
    return matched_faces


async def save_image_async(image, path):
    logger.debug("Saving image asynchronously to %s", path)
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format="JPEG")
    async with aiofiles.open(path, "wb") as f:
        await f.write(img_byte_arr.getvalue())
    logger.info("Image saved to %s", path)


def plot_face_info(image, matched_faces):
    """Annotates the given PIL image with bounding boxes, age, and gender."""
    logger.debug("Annotating image with face information.")
    draw = ImageDraw.Draw(image)
    width, height = image.size

    for face in matched_faces:
        bounding_box = face["bounding_box"]
        x_min, y_min, x_max, y_max = bounding_box["x_min"], bounding_box["y_min"], bounding_box["x_max"], bounding_box[
            "y_max"]

        draw.rectangle([x_min, y_min, x_max, y_max], outline="red", width=3)

        text = f"Age: {face.get('age', 'Unknown')}, Gender: {face.get('gender', 'Unknown')}"
        font = ImageFont.truetype('Roboto-Regular.ttf', size=20)
        draw.text((x_min, y_min - 30), text, font=font, fill="red")

        # Draw landmarks if available
        if "landmarks" in face:
            for point in face["landmarks"]:
                x = point["x"] * width
                y = point["y"] * height
                draw.ellipse([x - 2, y - 2, x + 2, y + 2], fill="blue")

    logger.debug("Image annotated successfully.")
    return image


class DataStorageService(aggregator_pb2_grpc.DataStorageServicer):
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))
        logger.info("Redis client initialized.")

    async def SaveFaceAttributes(self, request, context):
        redis_key = request.redis_key
        image_data = request.frame
        logger.info("Received SaveFaceAttributes request with redis_key: %s", redis_key)

        if not self.redis_client:
            logger.error("Redis client not initialized.")
            return aggregator_pb2.FaceResultResponse(response=False)

        landmarks_data = await self.redis_client.hget(redis_key, "landmarks")
        age_gender_data = await self.redis_client.hget(redis_key, "age_gender_results")

        if not landmarks_data or not age_gender_data:
            logger.warning("Missing data for redis_key: %s", redis_key)
            return aggregator_pb2.FaceResultResponse(response=False)

        try:
            landmarks = json.loads(landmarks_data)
            age_gender_results = json.loads(age_gender_data)
        except json.JSONDecodeError:
            logger.error("JSON decoding error for redis_key: %s", redis_key)
            return aggregator_pb2.FaceResultResponse(response=False)

        matched_faces = await match_faces_with_hungarian(landmarks, age_gender_results)

        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        image_filename = os.path.join(output_dir, f"output_{redis_key}.jpg")
        json_filename = os.path.join(output_dir, f"output_{redis_key}.json")
        annotated_filename = os.path.join(output_dir, f"annotated_{redis_key}.jpg")

        try:
            image = Image.open(io.BytesIO(image_data))
        except Exception as e:
            logger.error("Error opening image for redis_key: %s. Error: %s", redis_key, str(e))
            return aggregator_pb2.FaceResultResponse(response=False)

        await save_image_async(image, image_filename)

        async with aiofiles.open(json_filename, "w") as json_file:
            await json_file.write(json.dumps(matched_faces))

        annotated_image = plot_face_info(image.copy(), matched_faces)

        await save_image_async(annotated_image, annotated_filename)

        logger.info("Saved files: %s, %s, %s", image_filename, json_filename, annotated_filename)

        return aggregator_pb2.FaceResultResponse(response=True)


async def serve():
    server = grpc.aio.server(options=[('grpc.max_metadata_size', 1024 * 1024)])
    aggregator_pb2_grpc.add_DataStorageServicer_to_server(DataStorageService(), server)
    server.add_insecure_port("[::]:50053")
    logger.info("DataStorageService running on port 50053...")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    logger.info("Starting server...")
    asyncio.run(serve())
