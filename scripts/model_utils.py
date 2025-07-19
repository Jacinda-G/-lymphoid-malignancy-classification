# scripts/model_utils.py
import os
import torch
import urllib.request
import torchvision.models as models
import torch.nn as nn

def load_model(device="cpu"):
    model_path = "data/models/trained_resnet50.pth"
    os.makedirs(os.path.dirname(model_path), exist_ok=True)

    if not os.path.exists(model_path):
        url = "https://lymphomamlws4085132117.blob.core.windows.net/models/trained_resnet50.pth"
        urllib.request.urlretrieve(url, model_path)

    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, 3)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model
