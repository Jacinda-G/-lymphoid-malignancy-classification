# --- Imports ---
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import sys
from PIL import Image
import requests
from io import BytesIO
sys.path.append("./scripts")
from gradcam_utils import GradCAM
from scripts.evaluate import predict_tile  # moved to top

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tabs Navigation ---
tab_demo, tab_interactive = st.tabs(["🧪 Demo Mode", "🔍 Interactive Mode"])

# --- DEMO MODE ---
with tab_demo:
    st.title("Demo Mode")

    if "demo_step" not in st.session_state:
        st.session_state.demo_step = 0

    demo_sections = [
        "Pipeline Overview",
        "View Raw Data",
        "Tile Images",
        "Training Results",
        "GradCAM Visualization",
        "Model Evaluation",
    ]

    step = st.session_state.demo_step
    st.header(f"🧪 {demo_sections[step]}")

    if step == 0:
        st.write("This deep learning pipeline classifies hematological malignancies using whole slide images (WSIs).")
        st.image("demo_assets/pipeline_flowchart.png")
    elif step == 1:
        st.write("Sample WSIs from a public Kaggle dataset.")
        st.image("notebooks/demo_assets/sample_raw1.png", width=300)
    elif step == 2:
        st.write("Tiles (224x224) are extracted from WSIs for training.")
        st.image("notebooks/demo_assets/sample_tile1.png", width=150)
    elif step == 3:
        st.write("Model training was performed using a ResNet50 architecture.")
        st.write("Training included augmentation and optimization.")
        st.image("notebooks/demo_assets/sample_processed_tile1.png")
        st.write("Accuracy: 92%, F1 Score: 0.90")
    elif step == 4:
        st.write("GradCAM visualizes prediction-relevant regions.")
        st.image("notebooks/demo_assets/sample_gradcam_output.png")
    elif step == 5:
        st.write("Evaluation metrics include accuracy, precision, recall, and F1 score.")
        st.subheader("Confusion Matrix")
        st.image("demo_assets/confusion_matrix.png", caption="Confusion Matrix", use_container_width=True)
        st.subheader("ROC AUC Curve")
        st.image("demo_assets/roc_auc_curve.png", caption="ROC AUC Curve", use_container_width=True)

    col1, col2 = st.columns([1, 3])
    if col1.button("⬅ Back", disabled=(step == 0)):
        st.session_state.demo_step = max(step - 1, 0)
    if col2.button("Next ➡", disabled=(step == len(demo_sections) - 1)):
        st.session_state.demo_step = min(step + 1, len(demo_sections) - 1)

# --- INTERACTIVE MODE ---
with tab_interactive:
    st.title("Interactive Inference & Visualization")

    uploaded_tile = st.file_uploader("Upload a tile image (.png, .jpg, .tif)", type=["png", "jpg", "tif"])
    url = st.text_input("Or paste image URL (optional)")
    img = None

    if uploaded_tile is not None:
        img = Image.open(uploaded_tile).convert("RGB")
        st.image(img, caption="Uploaded Image", use_container_width=True)
    elif url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(img, caption="Image from URL", use_container_width=True)
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")

    if img and st.button("Classify & Explain"):
        with st.spinner("🔎 Running inference and generating GradCAM..."):
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 3)
            model.load_state_dict(torch.load("data/models/trained_resnet50.pth", map_location=device))
            model = model.to(device)
            model.eval()

            class_labels = ["CLL", "FL", "MCL"]
            predicted_label, confidence, _ = predict_tile(model, input_tensor, class_labels, device)

            st.success(f"🧠 **Prediction**: {predicted_label} ({confidence*100:.2f}% confidence)")

            try:
                target_layer = model.layer4[1].conv2
                gradcam = GradCAM(model, target_layer)
                heatmap = gradcam.generate(input_tensor)

                img_np = np.array(img.resize((224, 224)))
                heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                superimposed_img = heatmap_color * 0.4 + img_np

                st.image(superimposed_img.astype(np.uint8), caption="Grad-CAM Heatmap", use_container_width=True)
            except Exception as e:
                st.warning(f"GradCAM generation failed: {e}")
