import gradio as gr
import numpy as np
import requests
import json
import pandas as pd
from PIL import Image
import io
import base64
import tempfile

API_URL = "https://api-inference.huggingface.co/models/AliGhiasvand86/gisha_digit_recognition"
headers = {"Authorization": "Bearer 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"}

def query(image_path):
    with open(image_path, "rb") as file:
        response = requests.post(API_URL, headers=headers, files={"file": file})
    return response.json()

def predict(img):
    # Convert the numpy array to a PIL Image object
    img_pil = Image.fromarray((img * 255).astype(np.uint8))

    # Save the image to a temporary file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as temp_file:
        image_path = temp_file.name
        img_pil.save(image_path)

    # Send the image file path to the query function
    pred = query(image_path)

    # Convert the JSON output to a pandas DataFrame and return it
    df = pd.DataFrame(pred)
    return df

iface = gr.Interface(predict, inputs='sketchpad',
                     outputs=gr.outputs.Dataframe(),
                     allow_flagging='never',
                     description='Draw a Digit Below... (Draw in the centre for best results)')
iface.launch(share=False, width=300, height=500)
