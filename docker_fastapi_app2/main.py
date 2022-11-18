import io
import json
from PIL import Image
import numpy as np
import cv2
import torch
from fastapi import File, FastAPI
from fastapi.responses import Response, HTMLResponse

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
    
#create your API
app = FastAPI()

def image_to_byte_array(image: Image) -> bytes:
    """Convert a PIL.Image instance to bytes

    The source code is borrowed from Nori's solution:
    https://stackoverflow.com/questions/33101935/convert-pil-image-to-byte-array
    """
    imgByteArr = io.BytesIO()
    image.save(imgByteArr, format='jpeg')
    imgByteArr = imgByteArr.getvalue()
    return imgByteArr

def overlay_bboxes(image: Image, model_outputs):
    """Overlay bboxes onto the image"""

    model_outputs = model_outputs.pandas().xyxy[0]
    image = np.array(image)
    for idx in range(len(model_outputs)):
        x0, y0, x1, y1 = model_outputs.iloc[idx][:4].apply(lambda x: int(x))
        cv2.rectangle(image, (x0, y0), (x1, y1), (18, 127, 15), thickness=2)
    return Image.fromarray(image)

def concat_two_images(image1, image2, mode='h'):
    """Concatenate two PIL.Images instances"""

    if image1.width != image2.width or image1.height != image2.height:
        raise ValueError

    if mode == 'h':
        dst = Image.new('RGB', (image1.width + image2.width, image1.height))
        dst.paste(image1, (0, 0))
        dst.paste(image2, (image1.width, 0))
    elif mode == 'v':
        dst = Image.new('RGB', (image1.width, image1.height + image2.height))
        dst.paste(image1, (0, 0))
        dst.paste(image2, (0, image1.height))
    return dst

@app.post("/DetectionResults/")
async def get_body(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = model(input_image)
    results_json = json.loads(results.pandas().xyxy[0].to_json(orient="records"))
    output_arr = cv2.transpose(np.array(input_image))
    output_file = image_to_byte_array(Image.fromarray(output_arr))
    return {"result": results_json}

@app.post("/DetectionVisualization/")
async def get_body(file: bytes = File(...)):
    input_image = Image.open(io.BytesIO(file)).convert("RGB")
    results = model(input_image)
    output_image = overlay_bboxes(input_image, results)
    merged_image = concat_two_images(input_image, output_image)
    merged_file = image_to_byte_array(merged_image)
    return Response(merged_file, media_type="image/jpg")

@app.get("/")
async def root():
    content = """
<body>
<form action="/DetectionVisualization/" enctype="multipart/form-data" method="post">
<input name="file" type="file" multiple>
<input type="submit">
</form>
    """
    return HTMLResponse(content=content)

if __name__ == '__main__':
    img_path = 'dogs.jpeg'
    input_image = Image.open(img_path)
    results = model(input_image)
    output_image = overlay_bboxes(input_image, results)
