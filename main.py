import io
import base64

import numpy as np

from PIL import Image

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.meta_analysis import get_scene_info, get_image_classification
from src.llm_analysis import get_llm_analysis


def convert_np_to_base64(image_array):
    pil_img = Image.fromarray(image_array, 'RGB')
    
    buffered = io.BytesIO()
    pil_img.save(buffered, format="JPEG")
    img_bytes = base64.b64encode(buffered.getvalue())

    img_string = img_bytes.decode()
    
    return img_string

def predict(query: str, image_array):    
    image_classification = get_image_classification(image_array)
    image_description = get_scene_info(image_array)
    
    base64_image = convert_np_to_base64(image_array)
    
    response = get_llm_analysis(
        base64_image=base64_image,
        query=query,
        label=image_classification,
        image_description=image_description
    )
    
    return response

image = st.file_uploader(label="Upload the Image", type=["jpg", "jpeg", "png"])
if image:
    st.image(image)
    
query = st.text_input(label="query")
button = st.button("Submit")

if image is not None and button:
    image = Image.open(image)
    image_array = np.array(image)
    response = predict(query, image_array)
    
    st.text(response)
