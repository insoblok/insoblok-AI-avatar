# Start from a BARE NVIDIA CUDA image, not a full PyTorch one.
# This gives us a clean slate.
FROM nvidia/cuda:12.1.1-runtime-ubuntu22.04

# Set up the working environment
WORKDIR /app

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python and pip from the OS package manager
RUN apt-get update && apt-get install -y python3 python3-pip

# Copy our requirements file
COPY requirements.txt .

# Install our pinned dependencies. This is now a clean install.
RUN python3 -m pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY ./main.py .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]