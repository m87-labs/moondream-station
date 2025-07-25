from PIL import Image
import os
import requests
from io import BytesIO

def load_image(image_path: str) -> Image.Image:
   """Load an image from a file path or URL."""
   if image_path.startswith(('http://', 'https://')):
       response = requests.get(image_path, timeout=10)
       response.raise_for_status()
       return Image.open(BytesIO(response.content)).convert("RGB")
   
   if not os.path.exists(image_path):
       raise FileNotFoundError(f"Image not found at {image_path}")
   
   return Image.open(image_path).convert("RGB")