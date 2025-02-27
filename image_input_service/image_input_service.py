import grpc
import asyncio
import os
import glob
import aggregator_pb2
import aggregator_pb2_grpc
import logging

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


async def send_image_to_services(image_path):
    """Send image data to face landmark and age/gender services."""
    logger.info(f"Sending image {image_path} to services.")

    # Read the image data
    try:
        with open(image_path, "rb") as image_file:
            image_data = image_file.read()
        logger.debug(f"Read {len(image_data)} bytes from image {image_path}.")
    except Exception as e:
        logger.error(f"Failed to read image {image_path}: {e}")
        raise

    grpc_options = [('grpc.max_metadata_size', 1024 * 1024)]

    # Send image data to both services concurrently
    try:
        async with grpc.aio.insecure_channel('face_landmark_service:50051', options=grpc_options) as channel_1, \
                grpc.aio.insecure_channel('age_gender_service:50052', options=grpc_options) as channel_2:
            stub_1 = aggregator_pb2_grpc.FaceLandmarkDetectionStub(channel_1)
            stub_2 = aggregator_pb2_grpc.AgeGenderEstimationStub(channel_2)

            request = aggregator_pb2.ImageRequest(image_data=image_data)

            logger.info(f"Sending image {image_path} to face landmark service.")
            task_1 = stub_1.DetectLandmarks(request)

            logger.info(f"Sending image {image_path} to age/gender estimation service.")
            task_2 = stub_2.EstimateAgeGender(request)

            # Await both tasks to complete
            await asyncio.gather(task_1, task_2)

            logger.info(f"Successfully sent image {image_path} to both services.")
    except grpc.RpcError as e:
        logger.error(f"gRPC error while sending image {image_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error while sending image {image_path}: {e}")
        raise


async def process_directory(directory_path):
    """Process all image files in the given directory."""
    logger.info(f"Processing images in directory {directory_path}.")

    image_files = glob.glob(os.path.join(directory_path, "*.jpg"))
    if not image_files:
        logger.warning(f"No image files found in directory {directory_path}.")

    for image_path in image_files:
        logger.info(f"Processing image {image_path}.")
        await send_image_to_services(image_path)


if __name__ == "__main__":
    directory_path = "data"
    logger.info(f"Starting the image processing for directory {directory_path}.")
    asyncio.run(process_directory(directory_path))
    logger.info("Image processing completed.")
