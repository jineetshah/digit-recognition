import gradio as gr
import numpy as np
import requests
import json
import pandas as pd
from PIL import Image
import io
import base64

API_URL = "https://api-inference.huggingface.co/models/AliGhiasvand86/gisha_digit_recognition"
headers = {"Authorization": "Bearer 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"}

def query(data):
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()

def predict(img):
    # Convert the numpy array to a PIL Image object
    img_pil = Image.fromarray((img * 255).astype(np.uint8))

    # Convert the image data to a byte array
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    # Send the byte array as data in the query function
    pred = query(img_byte_arr)

    # Convert the JSON output to a pandas DataFrame and return it
    df = pd.DataFrame(pred)
    return df

iface = gr.Interface(predict, inputs = 'sketchpad',
                     outputs = gr.outputs.Dataframe(),
                     allow_flagging = 'never',
                     description = 'Draw a Digit Below... (Draw in the centre for best results)')
iface.launch(share = False, width = 300, height = 500)
