import gradio as gr
import numpy as np
import requests

API_URL = "https://api-inference.huggingface.co/models/AliGhiasvand86/gisha_digit_recognition"
headers = {"Authorization": "Bearer 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"}

def query(data):
    response = requests.post(API_URL, headers=headers, data=data)
    return response.json()


def predict(img):
    # img_array = np.array(img)
    # img_array = img_array.reshape(1, 28, 28)
    # img_array = img_array/255
    # pred = loaded_CNN.predict(img_array)/
    pred = query(img)
    return pred

iface = gr.Interface(predict, inputs = 'sketchpad',
                     outputs = 'text',
                     allow_flagging = 'never',
                     description = 'Draw a Digit Below... (Draw in the centre for best results)')
iface.launch(share = True, width = 300, height = 500)