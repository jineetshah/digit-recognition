import gradio as gr
import requests
import pandas as pd
from PIL import Image
import numpy as np
import base64

API_URL = "https://api-inference.huggingface.co/models/AliGhiasvand86/gisha_digit_recognition"
headers = {"Authorization": "Bearer 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"}

def query(image_path):
    with open(image_path, "rb") as file:
        response = requests.post(API_URL, headers=headers, files={"file": file})
    return response.json()

def save_array_as_image(array, image_path):
    # Convert the array to an image
    image = Image.fromarray(array)
    
    # Save the image to the specified path
    image.save(image_path)

def classify_digit(image):
    # Save the image as a .png file
    image_path = "sketchpad.png"
    save_array_as_image(image, image_path)
    
    result = query(image_path)
    df = pd.DataFrame(result)
    return df

iface = gr.Interface(fn=classify_digit, inputs='sketchpad', outputs=gr.outputs.Dataframe(),
                     allow_flagging='never', description='Draw a Digit Below... (Draw in the centre for best results)')
iface.launch(share=False, width=300, height=500)
