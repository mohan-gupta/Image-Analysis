from PIL import Image
import torch
from transformers import BlipProcessor, BlipForConditionalGeneration, AutoProcessor, AutoModel


clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base", use_fast=False)

def get_scene_info(image_arr):
    """
    Uses Blip Model for image captioning
    """
    image = Image.fromarray(image_arr).convert('RGB')
    
    inputs = blip_processor(image, return_tensors="pt")

    out = blip_model.generate(**inputs)
    response = blip_processor.decode(out[0], skip_special_tokens=True)
    
    return response

def get_image_classification(image_arr):
    """
    Uses CLIP model to 
    """
    labels = ["a photo of a cat", "a photo of a dog", "a photo of a car"]
    
    image = Image.fromarray(image_arr).convert('RGB')
    
    inputs = clip_processor(text=labels, images=image, return_tensors="pt", padding=True)

    outputs = clip_model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    most_likely_idx = probs.argmax(dim=1).item()
    most_likely_label = labels[most_likely_idx]
    
    return most_likely_label


if __name__ == "__main__":
    import numpy as np
    image = Image.open("../data/image.jpg")
    
    image_array = np.array(image)
    
    response = get_image_classification(image_array)
    print(response)