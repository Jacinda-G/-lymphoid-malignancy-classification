import os
import torch
import urllib.request
import torchvision.models as models
import torch.nn as nn

def load_model(device="cpu"):
    model_path = "data/models/trained_resnet50.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        url = "https://lymphomamlws4085132117.blob.core.windows.net/models/trained_resnet50.pth?sp=r&st=2025-07-22T02:27:58Z&se=2025-07-22T10:42:58Z&spr=https&sv=2024-11-04&sr=b&sig=O6dBgtZBSPbWEpabfDaN%2F34%2BEfrHNlvC6krUzwkG%2B5A%3D"
        try:
            print("📥 Downloading model from Azure Blob Storage...")
            urllib.request.urlretrieve(url, model_path)
            print("✅ Model downloaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to download model: {e}")

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)  # Adjust for 3 lymphoma classes
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    model.eval()
    return model
