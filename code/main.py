from fastapi import FastAPI, Request
from fastapi.responses import Response
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
from controlnet_aux import CannyDetector
import io
import traceback # Import for better error logging

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
print("Models loaded successfully.")
# --- End of model loading ---

app = FastAPI()

# This is a health check endpoint that SageMaker uses
@app.get("/ping")
def ping():
    return Response(status_code=200)

# This is the main inference endpoint
@app.post("/invocations")
async def invocations(request: Request):
    try:
        # Get the raw image bytes directly from the request body
        image_bytes = await request.body()
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # Process for ControlNet
        control_image = canny_detector(input_image, 100, 200)

        # The prompt is now hardcoded in the server
        prompt = "Notion style avatar, minimalist, vector art, black and white, clean lines, simple, corporate headshot"
        negative_prompt = "ugly, disfigured, deformed, noisy, blurry, low quality, watermark, text"

        # Generate the image
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

        # Return the image as a PNG response
        return Response(content=img_byte_arr.getvalue(), media_type="image/png")

    except Exception as e:
        # Log the full error to CloudWatch for debugging
        print(f"ERROR: {e}")
        traceback.print_exc()
        return Response(content=str(e), status_code=500)