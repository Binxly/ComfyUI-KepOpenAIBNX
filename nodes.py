from typing import Tuple
import os
import base64
import hashlib

import torch
import PIL
import numpy as np
from PIL import Image
from openai import Client as OpenAIClient

# Functions from image.py
def tensor2pil(image: torch.Tensor) -> PIL.Image.Image:
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def pil2base64(image: PIL.Image.Image) -> str:
    from io import BytesIO
    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

# Function from credentials.py
def get_open_ai_api_key() -> str:
    return os.environ.get("OPEN_AI_API_KEY", None)

class ImageWithPrompt:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "Image": ("IMAGE", {}),
                "prompt": (
                    "STRING",
                    {
                        "multiline": True,
                        "default": "I am going to give you an image, and i want you to describe them in approximately 256 tokens of exquisite detail. Generate a high quality caption for the image. The most important aspects of the image should be described first. If needed, weights can be applied to the caption in the following format: '(word or phrase:weight)', where the weight should be a float less than 2. Only reply with your proposed caption.",
                    },
                ),
                "max_tokens": ("INT", {"min": 128, "max": 2048, "default": 256}),
            }
        }

    RETURN_TYPES = ("STRING",)
    FUNCTION = "generate_completion"

    CATEGORY = "GPT-Interrogator"

    def __init__(self):
        self.open_ai_client: OpenAIClient = OpenAIClient(
            api_key=get_open_ai_api_key()
        )
        self.cache = {}

    def generate_completion(
        self, Image: torch.Tensor, prompt: str, max_tokens: int
    ) -> Tuple[str]:
        # Generate a cache key based on the image content and prompt
        image_hash = hashlib.md5(Image.cpu().numpy().tobytes()).hexdigest()
        cache_key = f"{image_hash}_{prompt}_{max_tokens}"

        # Check if the result is already in the cache
        if cache_key in self.cache:
            return (self.cache[cache_key],)

        # If not in cache, proceed with the API call
        b64image = pil2base64(tensor2pil(Image))
        response = self.open_ai_client.chat.completions.create(
            model="gpt-4o-mini",
            max_tokens=max_tokens,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64image}"},
                        },
                    ],
                }
            ],
        )
        if len(response.choices) == 0:
            raise Exception("No response from OpenAI API")

        result = response.choices[0].message.content
        
        # Store the result in the cache
        self.cache[cache_key] = result

        return (result,)