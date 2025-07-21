from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import StreamingResponse
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import io

# --- Initialize models (this happens once when the app starts) ---
print("Loading models...")
controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype=torch.float16)
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5", 
    controlnet=controlnet, 
    torch_dtype=torch.float16
)
pipe.to("cuda")
pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
canny_detector = CannyDetector()
print("Models loaded.")
# --- End of model loading ---

app = FastAPI()

@app.get("/")
def read_root():
    return {"Status": "API is running"}

@app.post("/generate-avatar/")
async def generate_avatar(
    file: UploadFile = File(...),
    prompt: str = Form("Notion style avatar, minimalist, vector art, black and white, clean lines")
):
    # Read the uploaded image file
    image_bytes = await file.read()
    input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    # Process for ControlNet
    low_threshold = 100
    high_threshold = 200
    control_image = canny_detector(input_image, low_threshold, high_threshold)

    # Generate the image
    negative_prompt = "ugly, disfigured, deformed, noisy, blurry, low quality"
    generated_image = pipe(
        prompt,
        negative_prompt=negative_prompt,
        image=control_image,
        num_inference_steps=20,
    ).images[0]

    # Save the generated image to a byte stream
    img_byte_arr = io.BytesIO()
    generated_image.save(img_byte_arr, format='PNG')
    img_byte_arr.seek(0)

    return StreamingResponse(img_byte_arr, media_type="image/png")