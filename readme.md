
## Face Detection System

### Overview

This repository provides a **Face Detection System** that utilizes multiple services to perform face-related tasks such as detecting facial landmarks, predicting age and gender, and aggregating the results into structured output.

The system consists of multiple microservices, including:

1. **Image Input Service** – Receives and processes input images.
2. **Face Landmark Service** – Detects facial landmarks from the image.
3. **Age/Gender Service** – Predicts the age and gender of the faces detected in the image.
4. **Data Storage Service** – Stores the output results (image and JSON) and presents them through a web interface.

The output of the system will be stored in the `output` directory at the root of the project.

### Setup

#### Prerequisites

Ensure you have the following installed:
- Docker
- Docker Compose

#### Steps to Set Up

1. Clone the repository:

    ```bash
    git clone https://github.com/yazdanifar/face-detection.git
    cd face-detection
    ```

2. **Place the Data**: Place your images in the `data` directory at the root of the project. This directory should contain the images you want to process (e.g., `MultipleFaces.jpg`, `SingleFace.jpg`).

3. **Run the Services with Docker Compose**: Use Docker Compose to start all the necessary services and dependencies (e.g., Redis). In the project directory, run the following command:

    ```bash
    docker-compose up --build
    ```

    This will build and start all the services defined in the `docker-compose.yml` file.

4. **Output Directory**: After processing, the output will be located in the `output` directory in the project root. This directory will contain images and corresponding JSON files with the results.

### Architecture

The Face Detection System uses **asyncio** to run multiple tasks concurrently, ensuring efficient processing of the images.

#### Key Services and Data Flow

1. **Image Input Service**:
   - Receives an image, processes it, and forwards the image data to both the **Face Landmark Service** and **Age/Gender Service**.
   
2. **Face Landmark Service**:
   - Detects key facial landmarks (eyes, nose, mouth, etc.) in the image and stores the bounding boxes of the detected faces in Redis, along with the landmark coordinates.
   
3. **Age/Gender Service**:
   - Predicts the age and gender of each face detected in the image. The bounding boxes of the faces are also stored in Redis, along with the predicted age and gender.

4. **Data Storage Service**:
   - This service pulls data from Redis, aggregates the results (by using the **Hungarian Algorithm**), and stores the final output (image and JSON data) in the `output` directory. The data includes the face landmarks, age, gender, and bounding box details.

### Hungarian Algorithm for Result Aggregation

In this system, the **Hungarian Algorithm** is used to match bounding boxes from the **Face Landmark Service** and the **Age/Gender Service**.

#### Why Use the Hungarian Algorithm?

The **Hungarian Algorithm** is used to solve the **assignment problem**, where we want to optimally match two sets of items based on a cost function. In this case, the two sets are:

1. **Bounding Boxes from Face Landmark Service** – Enclose faces and include landmark data.
2. **Bounding Boxes from Age/Gender Service** – Enclose faces and include predicted age and gender.

The bounding boxes from both services may differ in terms of order, size, and positioning. The Hungarian Algorithm helps match these bounding boxes in an optimal way, based on a **cost function** that minimizes the mismatch between the bounding boxes.

#### How the Hungarian Algorithm Works in This System

1. **Cost Matrix Construction**:
   - A cost matrix is created, where each entry represents the **1 - IoU (Intersection over Union)** between a bounding box from the **Face Landmark Service** and a bounding box from the **Age/Gender Service**.
   
     **IoU Calculation**:
     - **IoU** is a measure used to evaluate the overlap between two bounding boxes. It is calculated as the area of overlap between the two bounding boxes divided by the area of their union.
     - **1 - IoU** is used as the cost because the algorithm seeks to minimize the cost (maximize overlap). A higher overlap results in a smaller cost.

2. **Hungarian Algorithm Application**:
   - The Hungarian Algorithm is applied to the cost matrix to find the optimal pairing of bounding boxes. The algorithm finds the assignment that minimizes the total cost (i.e., maximizes the IoU), ensuring that the correct face landmarks are paired with the corresponding age and gender predictions.
   
3. **Result Aggregation**:
   - After the algorithm runs, the bounding boxes are matched, and the results from the **Face Landmark Service** and **Age/Gender Service** are aggregated. The final output includes the face landmarks, predicted age, gender, and the bounding box data.

4. **Optimization**:
   - The Hungarian Algorithm guarantees that the bounding boxes are matched in the most optimal way, ensuring accurate results for each face detected in the image.

#### Example of Matching Bounding Boxes

Let’s consider two services that detect faces:

- **Face Landmark Service** detects 3 faces and returns 3 bounding boxes.
- **Age/Gender Service** also detects 3 faces and returns 3 bounding boxes.

A **cost matrix** is created by calculating the **1 - IoU** between the bounding boxes from both services. The Hungarian Algorithm finds the optimal matching, ensuring that the face landmarks and age/gender predictions are correctly paired.

The matching minimizes the total **1 - IoU** cost, ensuring that each bounding box from the **Face Landmark Service** is matched with the corresponding one from the **Age/Gender Service**.

### Output Format

The output consists of two files for each processed image:

1. **Image file**: An image with the detected faces and bounding boxes annotated.
2. **JSON file**: A JSON file containing the following details:
   - Bounding box coordinates
   - Face landmarks (coordinates of eyes, nose, mouth, etc.)
   - Predicted age and gender
   - Other metadata (if applicable)

These files are stored in the `output` directory.
