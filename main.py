from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers import StableDiffusionPipeline
import torch
import warnings
import uvicorn
import io
from starlette.responses import StreamingResponse
import os
from fastapi import FastAPI, Path
from pydantic import BaseModel
from logger import log
# Setting for no warning messages. 
warnings.filterwarnings("ignore")
app = FastAPI()

from svdiff_pytorch import load_unet_for_svdiff, load_text_encoder_for_svdiff

num_inference_steps= int(os.getenv('num_inference_steps')) or 2

pretrained_model_name_or_path = "runwayml/stable-diffusion-v1-5"
spectral_shifts_ckpt_dir = None
unet = load_unet_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=spectral_shifts_ckpt_dir, subfolder="unet")
text_encoder = load_text_encoder_for_svdiff(pretrained_model_name_or_path, spectral_shifts_ckpt=spectral_shifts_ckpt_dir, subfolder="text_encoder")
pipe = StableDiffusionPipeline.from_pretrained(
    pretrained_model_name_or_path,
    unet=unet,
    text_encoder=text_encoder,
)
pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
#pipe.to("cuda")

class Request(BaseModel):
    text: str

def process_text(text):
    image = pipe(text, num_inference_steps=num_inference_steps).images[0]
    return image


@app.post("/predict_image")
def image_endpoint1(request: Request):
    text = request.text
    log.info(f"processing text:::{text}")
    img = process_text(text)
    img.save('img.jpg')
    return StreamingResponse(io.BytesIO(img.tobytes()), media_type="image/png")

@app.post("/predict_image/byteString")
def image_endpoint2(request: Request):
    text = request.text
    # Returns a cv2 image array from the document vector
    log.info(f"processing text:::{text}")
    img = process_text(text)
    # log.info("saving image")
    img.save('out.jpg')
    # log.info("image saved")
    arr = io.BytesIO()
    img.save(arr, format='PNG')
    # log.info("Saving array")
    img_arr = arr.getvalue()
    # log.info(img_arr)
    return str(img_arr)


if __name__ == '__main__':
	uvicorn.run(app,host='0.0.0.0', port=3000)