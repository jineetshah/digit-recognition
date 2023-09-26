import gradio as gr
import requests
import pandas as pd

API_URL = "https://api-inference.huggingface.co/models/AliGhiasvand86/gisha_digit_recognition"
headers = {"Authorization": "Bearer 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"}

def query(image_path):
    with open(image_path, "rb") as file:
        response = requests.post(API_URL, headers=headers, files={"file": file})
    return response.json()

def classify_digit(image):
    result = query(image.name)
    df = pd.DataFrame(result)
    return df

sketchpad = gr.inputs.Sketchpad()
output_box = gr.outputs.Dataframe()

iface = gr.Interface(fn=classify_digit, inputs=sketchpad, outputs=output_box)
iface.launch()
