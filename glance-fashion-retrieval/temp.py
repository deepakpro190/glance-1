import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

img = Image.new("RGB", (224, 224), "white")

inputs = processor(images=img, return_tensors="pt")

with torch.no_grad():
    out = model.get_image_features(**inputs)

print(type(out))
print(out)

print("Has pooler_output:", hasattr(out, "pooler_output"))