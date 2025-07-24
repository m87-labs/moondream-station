import moondream as md
from PIL import Image


model = md.vl(endpoint="your-api-key")
image = Image.open("path/to/image.jpg")

result = model.query(image, reasoning=True, question="What's in this image?")
answer = result["answer"]
reasoning = result["reasoning"]
print(f"Answer: {answer}")
print(f"Request ID: {reasoning}")
