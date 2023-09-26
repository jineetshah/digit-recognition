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

iface = gr.Interface(fn=classify_digit, inputs='sketchpad', outputs=gr.outputs.Dataframe(),
                     allow_flagging='never', description='Draw a Digit Below... (Draw in the centre for best results)')
iface.launch(share=False, width=300, height=500)
