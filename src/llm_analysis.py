import os

from openai import OpenAI

client = OpenAI(
    api_key=os.getenv("GEMINI_API_KEY"),
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

def get_llm_analysis(base64_image, query: str, label, image_description):
    prompt = f"""
    Answer the user query for the provided image.
    You are also provided scene description and image label.
    
    image lable: {label}
    dscription: {image_description}
    
    user query: {query}
    """
    
    response = client.chat.completions.create(
    model="gemini-3-flash-preview",
    messages=[{
				"role": "user",
				"content": [
				{
					"type": "text",
					"text": prompt,
				},
				{
					"type": "image_url",
					"image_url": {
					"url":  f"data:image/jpeg;base64,{base64_image}"
					},
				},
				],
			}],
		)
    
    return response.choices[0].message.content