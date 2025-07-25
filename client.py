import boto3
import json

# --- Configuration ---
# 1. The name of the SageMaker endpoint you deployed. (This is correct)
ENDPOINT_NAME = 'insoblok-avatar-generator-endpoint'

# 2. The AWS region where your endpoint is deployed. (This is correct)
REGION_NAME = 'us-east-1'

# 3. The path to the local image you want to transform.
#    IMPORTANT: Change this to the actual name of your image file.
image_path = 'user_image.png' # <--- MAKE SURE THIS FILENAME IS CORRECT

# 4. The name for the output file that will be saved.
output_path = 'cloud_generated_avatar.png'
# --- End Configuration ---


print(f"Sending '{image_path}' to SageMaker endpoint '{ENDPOINT_NAME}'...")

# Create a SageMaker runtime client
sagemaker_runtime = boto3.client("sagemaker-runtime", region_name=REGION_NAME)

try:
    # Read the image file from your local disk in binary mode
    with open(image_path, "rb") as f:
        image_payload = f.read()

    # Determine the content type based on the file extension
    content_type = 'image/jpeg' if image_path.lower().endswith(('.jpg', '.jpeg')) else 'image/png'

    # Invoke the endpoint, sending the raw bytes of the image
    response = sagemaker_runtime.invoke_endpoint(
        EndpointName=ENDPOINT_NAME,
        ContentType=content_type,
        Body=image_payload
    )

    # The response Body is a StreamingBody object. We read the generated image from it.
    generated_image_bytes = response['Body'].read()

    # Save the returned image to a new file
    with open(output_path, 'wb') as f:
        f.write(generated_image_bytes)

    print(f"\n✅ Success! Avatar saved as '{output_path}'")
    print("Open the file to see your AI-generated avatar!")

except FileNotFoundError:
    print(f"\n❌ ERROR: Cannot find the file '{image_path}'.")
    print("Please make sure the image exists and the path is correct.")
except Exception as e:
    # This will catch errors from AWS, like if the endpoint has an issue
    print(f"\n❌ An error occurred: {e}")