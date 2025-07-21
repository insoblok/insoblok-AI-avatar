# InSoBlok AI Avatar Generator

This project provides a complete solution for building and deploying an AI model that transforms user photos into minimalist, InSoBlok-style avatars. It uses Stable Diffusion with ControlNet to preserve the user's pose and facial structure while completely changing the artistic style based on a text prompt.

The entire application is designed to be tested locally and then deployed as a scalable, production-ready API endpoint on AWS SageMaker.

## ✨ Features

*   **Pose Preservation:** Uses ControlNet with Canny Edges to ensure the generated avatar matches the composition of the input image.
*   **Customizable Style:** The artistic style is controlled by a simple text prompt, making it easy to tweak.
*   **Local API:** Includes a FastAPI server for easy local development and testing.
*   **Scalable Cloud Deployment:** Packaged with Docker and designed for deployment on AWS SageMaker for a robust, scalable inference endpoint.

## 🖼️ Example

The model takes a regular photo and converts it into a clean, black-and-white vector-style avatar, perfect for professional profiles.

| Original User Photo                                       | Generated Avatar                                                |
| :--------------------------------------------------------: | :--------------------------------------------------------------: |
| <img src="/user_image.png" width="300">                | <img src="insoblok_ai_avatar.png" width="300">                   |
| *Input: A standard user photograph.*                       | *Output: A minimalist avatar in the requested style.*           |
<!-- Note: You should create a 'docs' folder and place example images there for this to render correctly on GitHub -->


## ⚙️ Technology Stack

*   **AI Model:** Stable Diffusion 1.5 + ControlNet (Canny Edge)
*   **Python Libraries:** Hugging Face `diffusers`, `transformers`, `torch`
*   **API Framework:** FastAPI
*   **Containerization:** Docker
*   **Cloud Platform:** AWS (SageMaker for hosting, ECR for container registry)

---

## 🚀 Getting Started

Follow these steps to get the project running locally and then deploy it to the cloud.

### Prerequisites

*   Python 3.8+
*   An **NVIDIA GPU** with at least 8GB VRAM (required for reasonable performance)
*   [NVIDIA CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit-archive) installed
*   [Docker](https://www.docker.com/get-started/)
*   [AWS CLI](https://aws.amazon.com/cli/) installed and [configured](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-quickstart.html)

### Phase 1: Local Development & Testing

First, let's get the model running on your local machine.

**1. Clone the repository and set up the environment:**
```bash
git clone https://github.com/your-username/notion-avatar-generator.git
cd notion-avatar-generator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

**2. Install dependencies:**
```bash
pip install -r requirements.txt
```
*Note: The first time you run the scripts, they will download several gigabytes of model weights from Hugging Face.*

**3. Run a single image generation test:**
```bash
python generate.py
```
Place a test image (e.g., `user_image.jpg`) in the root directory. Then, update the path in `generate.py` and run it:
```bash
python generate.py
```
This will create `notion_avatar.png` in your project folder.

**4. Run the local API server:**
```bash
uvicorn main:app --reload
```
Navigate to **`http://127.0.0.1:8000/docs`** in your browser to access the interactive API documentation (Swagger UI) where you can upload an image and test the API endpoint.

---

### Phase 2: Cloud Deployment on AWS SageMaker

Now, we'll package the application and deploy it to a scalable AWS endpoint.

**1. Build and Push the Docker Container to ECR:**
First, create a repository in Amazon ECR. Replace `us-east-1` with your preferred region.
```bash
aws ecr create-repository --repository-name insoblok-avatar-generator --region us-east-1
```

Next, build the Docker image and push it. **Replace `123456789012` with your AWS Account ID.**
```bash
# Authenticate Docker with ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 123456789012.dkr.ecr.us-east-1.amazonaws.com

# Build the Docker image
docker build -t insoblok-avatar-generator .

# Tag the image for ECR
docker tag insoblok-avatar-generator:latest 123456789012.dkr.ecr.us-east-1.amazonaws.com/insoblok-avatar-generator:latest

# Push the image
docker push 123456789012.dkr.ecr.us-east-1.amazonaws.com/insoblok-avatar-generator:latest
```

**2. Deploy to a SageMaker Endpoint:**
Before running the script, you need to:
*   Create an IAM Role with `AmazonSageMakerFullAccess` permissions.
*   Update `deploy.py` with your **IAM Role ARN** and your **ECR Image URI**.

Then, run the deployment script:
```bash
pip install sagemaker boto3
python deploy.py
```
This process will take **10-15 minutes** as AWS provisions the infrastructure.

---

## 🔌 API Usage

### Calling the Deployed SageMaker Endpoint

You can use the `boto3` SDK in Python to invoke your live endpoint. The `client.py` script provides a working example.

Update `client.py` with your `ENDPOINT_NAME` and `REGION_NAME`, then run:
```python
import boto3

# Configuration
ENDPOINT_NAME = 'insoblok-avatar-generator-endpoint' # As defined in deploy.py
REGION_NAME = 'us-east-1' # Your AWS region

sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION_NAME)

# Send an image file to the endpoint
with open("path/to/your/user_image.jpg", "rb") as f:
    image_payload = f.read()

response = sagemaker_runtime.invoke_endpoint(
    EndpointName=ENDPOINT_NAME,
    ContentType='image/jpeg',
    Body=image_payload
)

# Save the returned avatar
generated_image_bytes = response['Body'].read()
with open('cloud_generated_avatar.png', 'wb') as f:
    f.write(generated_image_bytes)

print("Avatar saved successfully!")
```

## ⚠️ Important Considerations

### AWS Costs

GPU instances for SageMaker are billed by the hour and can be expensive. **Always remember to shut down your endpoint when not in use to avoid unnecessary costs.**

Use the AWS CLI to delete the endpoint:
```bash
aws sagemaker delete-endpoint --endpoint-name insoblok-avatar-generator-endpoint --region us-east-1
```
You should also delete the model and endpoint configuration from the SageMaker console.

### Cold Starts

The first request to an idle endpoint may take longer (1-2 minutes) as the container and model are loaded into memory. For production applications requiring low latency, consider enabling [SageMaker Provisioned Concurrency](https://docs.aws.amazon.com/sagemaker/latest/dg/provisioned-concurrency.html).

## 🔮 Future Improvements

*   **Fine-tuning with LoRA:** For a more consistent and unique style, the base model can be fine-tuned using LoRA (Low-Rank Adaptation) on a small dataset of example avatars.
*   **API Gateway Integration:** Place an Amazon API Gateway in front of the SageMaker endpoint to handle authentication (API keys), rate limiting, and caching.
*   **Web Frontend:** Build a simple web interface (e.g., using Streamlit or React) to allow users to easily upload their photos.

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.