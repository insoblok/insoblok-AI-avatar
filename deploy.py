import sagemaker
from sagemaker.huggingface import HuggingFaceModel
import boto3

# --- Configuration ---
# 1. Your IAM Role ARN (This is correct)
ROLE_ARN = 'arn:aws:iam::017116124664:role/SageMaker-Avatar-Generator-Role'

# 2. AWS Region (This is correct)
REGION_NAME = 'us-east-1'

# 3. SageMaker Instance Type
INSTANCE_TYPE = 'ml.g5.xlarge'

# 4. Name for your SageMaker Endpoint
ENDPOINT_NAME = 'insoblok-avatar-generator-endpoint'
# --- End Configuration ---

print("--- Starting AWS Native SageMaker Deployment ---")
print(f"Role ARN: {ROLE_ARN}")
print(f"Region: {REGION_NAME}")
print("-" * 50)

try:
    # This uses a pre-built, GUARANTEED-to-work container from AWS.
    # It will automatically install your requirements.txt from the 'code' folder.
    huggingface_model = HuggingFaceModel(
        source_dir='./code',      # path to the directory with your code
        entry_point='main.py',      # your FastAPI script
        role=ROLE_ARN,              # your IAM role
        transformers_version='4.37.0', # Compatible version
        pytorch_version='2.1.0',       # Compatible version
        py_version='py310',          # Python version
        sagemaker_session=sagemaker.Session()
    )

    # Deploy the model to an endpoint
    print(f"Deploying model to a '{INSTANCE_TYPE}' instance. This will take 10-15 minutes...")
    predictor = huggingface_model.deploy(
        initial_instance_count=1,
        instance_type=INSTANCE_TYPE,
        endpoint_name=ENDPOINT_NAME,
    )

    print("\n✅ --- DEPLOYMENT SUCCESSFUL! --- ✅")
    print(f"Your endpoint is now live and ready to be used.")
    print(f"Endpoint Name: {predictor.endpoint_name}")

except Exception as e:
    print("\n❌ --- DEPLOYMENT FAILED --- ❌")
    print("An error occurred during deployment. Please check the following:")
    print("1. Ensure the IAM Role ARN is correct and has SageMaker permissions.")
    print("2. Check the CloudWatch logs for the endpoint for more details.")
    print(f"\nError details: {e}")