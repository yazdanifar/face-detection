import aiofiles
import os
import json
import grpc
import redis.asyncio as redis
import numpy as np
from scipy.optimize import linear_sum_assignment
import asyncio

import aggregator_pb2
import aggregator_pb2_grpc


def iou(boxA, boxB):
    """Compute Intersection over Union (IoU) between two bounding boxes."""
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
    iou = inter_area / float(boxA_area + boxB_area - inter_area) if (boxA_area + boxB_area - inter_area) > 0 else 0
    return iou


async def match_faces_with_hungarian(landmarks, age_gender_results):
    """
    Match faces using the Hungarian algorithm based on IoU.
    """
    num_landmarks = len(landmarks)
    num_age_gender = len(age_gender_results)
    cost_matrix = np.zeros((num_landmarks, num_age_gender))

    for i, landmark in enumerate(landmarks):
        for j, age_gender in enumerate(age_gender_results):
            cost_matrix[i, j] = 1 - iou(landmark["bounding_box"], age_gender["bounding_box"])  # Use 1 - IoU as cost

    # Apply the Hungarian algorithm
    landmark_indices, age_gender_indices = linear_sum_assignment(cost_matrix)
    print(landmark_indices, age_gender_indices)
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

    return matched_faces


class DataStorageService(aggregator_pb2_grpc.DataStorageServicer):
    def __init__(self):
        self.redis_client = redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379"))

    async def SaveFaceAttributes(self, request, context):
        redis_key = request.redis_key
        image_data = request.frame

        if not self.redis_client:
            return aggregator_pb2.FaceResultResponse(response=False)

        landmarks_data = await self.redis_client.hget(redis_key, "landmarks")
        age_gender_data = await self.redis_client.hget(redis_key, "age_gender_results")

        if not landmarks_data or not age_gender_data:
            return aggregator_pb2.FaceResultResponse(response=False)

        try:
            landmarks = json.loads(landmarks_data)
            age_gender_results = json.loads(age_gender_data)
        except json.JSONDecodeError:
            return aggregator_pb2.FaceResultResponse(response=False)

        matched_faces = await match_faces_with_hungarian(landmarks, age_gender_results)

        # Create the output directory if it doesn't exist
        output_dir = "output"
        os.makedirs(output_dir, exist_ok=True)

        # Save the image file asynchronously inside the output directory
        image_filename = os.path.join(output_dir, f"output_{redis_key}.jpg")
        async with aiofiles.open(image_filename, "wb") as image_file:
            await image_file.write(image_data)
        print(f"Image saved as: {image_filename}")

        # Save the JSON result asynchronously inside the output directory
        output_filename = os.path.join(output_dir, f"output_{redis_key}.json")
        async with aiofiles.open(output_filename, "w") as f:
            await f.write(json.dumps(matched_faces))

        print(f"Saved: {output_filename}")

        return aggregator_pb2.FaceResultResponse(response=True)


async def serve():
    server = grpc.aio.server(options=[('grpc.max_metadata_size', 1024 * 1024)])
    aggregator_pb2_grpc.add_DataStorageServicer_to_server(DataStorageService(), server)
    server.add_insecure_port("[::]:50053")
    print("DataStorageService running on port 50053...")
    await server.start()
    await server.wait_for_termination()


if __name__ == "__main__":
    asyncio.run(serve())
