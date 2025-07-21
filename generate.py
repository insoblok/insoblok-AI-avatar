import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import cv2
import numpy as np

# --- 1. Load the models ---
# Use a pre-trained Canny edge detection model for ControlNet
controlnet = ControlNetModel.from_pretrained(
    "lllyasviel/sd-controlnet-canny", 
    torch_dtype=torch.float16
)

# Use a base Stable Diffusion model. v1.5 is a good starting point.
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
)

# --- 2. Optimize for performance ---
# Move models to GPU
pipe.to("cuda")
# Use a faster scheduler
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
# Enable model CPU offloading if you have less VRAM (optional, slower)
# pipe.enable_model_cpu_offload()

# --- 3. Prepare the input image ---
# Load your input image
input_image = Image.open("E:/Google Drive/Github/insoblok-ai/insoblok-ai-avatar/user_image.png")

# --- 4. Pre-process the image for ControlNet ---
# Initialize the Canny edge detector
canny_detector = CannyDetector()
# Convert PIL image to a numpy array for OpenCV processing
low_threshold = 100
high_threshold = 200
control_image = canny_detector(input_image, low_threshold, high_threshold)

# control_image is now a PIL Image of the detected edges
control_image.save("canny_edges.png") # You can save this to see what the model "sees"

# --- 5. Craft the Prompt ---
# This is where you define the "Avatar" style. Be descriptive!
prompt = "Notion style avatar, minimalist, vector art, black and white, clean lines, simple, corporate headshot"
negative_prompt = "ugly, disfigured, deformed, noisy, blurry, low quality, watermark, text"

# --- 6. Generate the image ---
print("Generating avatar...")
generated_image = pipe(
    prompt,
    negative_prompt=negative_prompt,
    image=control_image, # This is the edge map
    num_inference_steps=20,
).images[0]

# --- 7. Save the output ---
generated_image.save("insoblok_ai_avatar.png")
print("Avatar saved as insoblok_ai_avatar.png")