from fastapi import FastAPI, UploadFile
from evaluate import predict_tile
from torchvision import transforms
from PIL import Image
import torch
import io

app = FastAPI()

# Load model ONCE at startup
model = torch.load("data/models/trained_resnet50.pth", map_location="cpu")
class_labels = ["CLL", "FL", "MCL"]


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

@app.post("/predict")
async def predict(file: UploadFile):
    contents = await file.read()

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        tensor = transform(image).unsqueeze(0)  # Add batch dimension
        predicted_label, confidence, probs = predict_tile(model, tensor, class_labels=class_labels)
        return {
            "predicted_label": predicted_label,
            "confidence": round(confidence, 4),
            "probabilities": {label: round(float(p), 4) for label, p in zip(class_labels, probs.squeeze().tolist())}
        }
    except Exception as e:
        return {"error": str(e)}
