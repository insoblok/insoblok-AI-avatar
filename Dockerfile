# Use an official NVIDIA base image with PyTorch and CUDA pre-installed
FROM nvidia/pytorch:2.0.1-cuda11.8-cudnn8-runtime-ubuntu22.04

# Set up a working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
# Note: We don't specify the torch index-url here as the base image already has it.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code into the container
COPY ./main.py .

# Expose the port the app runs on (FastAPI default is 8000)
EXPOSE 8000

# Command to run the application using uvicorn
# We use 0.0.0.0 to allow traffic from outside the container
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]