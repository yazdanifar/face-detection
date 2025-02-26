import grpc
import asyncio
import os
import glob
import aggregator_pb2
import aggregator_pb2_grpc


async def send_image_to_services(image_path):
    with open(image_path, "rb") as image_file:
        image_data = image_file.read()

    # Set up two asynchronous tasks to send requests to both services
    grpc_options = [('grpc.max_metadata_size', 1024 * 1024)]
    async with grpc.aio.insecure_channel('face_landmark_service:50051', options=grpc_options) as channel_1, \
            grpc.aio.insecure_channel('age_gender_service:50052', options=grpc_options) as channel_2:
        # Prepare the stubs for both services
        stub_1 = aggregator_pb2_grpc.FaceLandmarkDetectionStub(channel_1)
        stub_2 = aggregator_pb2_grpc.AgeGenderEstimationStub(channel_2)

        # Create the requests
        request = aggregator_pb2.ImageRequest(image_data=image_data)

        # Fire-and-forget requests for both services
        task_1 = stub_1.DetectLandmarks(request)
        task_2 = stub_2.EstimateAgeGender(request)

        # Await both tasks concurrently
        await asyncio.gather(task_1, task_2)


async def process_directory(directory_path):
    # Find all .jpg files in the given directory
    image_files = glob.glob(os.path.join(directory_path, "*.jpg"))
    for image_path in image_files:
        await send_image_to_services(image_path)


if __name__ == "__main__":
    directory_path = "data"  # Replace with your directory path
    asyncio.run(process_directory(directory_path))
