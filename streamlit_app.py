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

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

uploaded_tile = None  # <-- Initialize early!
uploaded_files = None  

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["Demo Mode", "Interactive Mode"])

# --- Demo Step Setup ---
if mode == "Demo Mode":
    if "demo_step" not in st.session_state:
        st.session_state.demo_step = 0

    demo_sections = [
        "Pipeline Overview",
        "View Raw Data",
        "Tile Images",
        "Training Results",
        "GradCAM Visualization"
    ]

    step = st.session_state.demo_step
    st.header(f"🧪 {demo_sections[step]}")

    if step == 0:
        st.write("🔬 Overview of the entire deep learning pipeline...")
        st.image("demo_assets/pipeline_flowchart.png")
    elif step == 1:
        st.write("🧫 Sample Whole Slide Images (WSIs)...")
        st.image("notebooks/demo_assets/sample_raw1.png", width=300)
    elif step == 2:
        st.write("🧩 Example of a tiled image patch (224x224)...")
        st.image("notebooks/demo_assets/sample_tile1.png", width=150)
    elif step == 3:
        st.write("📈 Model accuracy and F1 Score...")
        st.image("notebooks/demo_assets/sample_processed_tile1.png")
        st.write("Accuracy: 92%, F1 Score: 0.90")
    elif step == 4:
        st.write("🔍 GradCAM highlights areas of diagnostic relevance...")
        st.image("notebooks/demo_assets/sample_gradcam_output.png")

    col1, col2 = st.columns([1, 3])
    if col1.button("⬅ Back", disabled=(step == 0)):
        st.session_state.demo_step = max(step - 1, 0)
    if col2.button("Next ➡", disabled=(step == len(demo_sections) - 1)):
        st.session_state.demo_step = min(step + 1, len(demo_sections) - 1)


# Go to interactive mode
if st.button("➡ Go to Interactive Mode"):
    st.session_state.mode = "Interactive Mode (Upload & Train)"

    interactive_section = st.sidebar.radio(
        "Interactive Actions", 
        ["Upload Data, Inference, & Visualize"]
    )

    uploaded_tile = None
    uploaded_files = None
    img = None  # For inference image

    # --- Inference & Visualize Section ---
# --- Interactive Mode (Single Page Version) ---
elif mode == "Interactive Mode (Upload & Train)":
    st.header("Upload, Inference, & Visualize")
    st.write("Upload a tile image or paste an image URL to classify and visualize model explanations using GradCAM.")

    uploaded_tile = st.file_uploader("Upload a tile image (.png, .jpg, .tif)", type=["png", "jpg", "tif"])
    url = st.text_input("Or paste image URL (optional)")

    img = None

    if uploaded_tile is not None:
        img = Image.open(uploaded_tile).convert("RGB")
        st.image(img, caption="Uploaded Image", use_column_width=True)

    elif url:
        try:
            response = requests.get(url)
            img = Image.open(BytesIO(response.content)).convert("RGB")
            st.image(img, caption="Image from URL", use_column_width=True)
        except Exception as e:
            st.error(f"Could not load image from URL: {e}")

    if img and st.button("Classify & Explain"):
        with st.spinner("🔎 Running inference and generating GradCAM..."):

            # Preprocess
            preprocess = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
            ])
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            # Load model
            model = models.resnet50(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 3)  
            model.load_state_dict(torch.load("data/models/trained_resnet50.pth", map_location=device))
            model = model.to(device)
            model.eval()

            # Predict
            # Call evaluate.py function
            class_labels = ["CLL", "FL", "MCL"]
            predicted_label, confidence, _ = predict_tile(model, input_tensor, class_labels, device)

            st.success(f"🧠 **Prediction**: {predicted_label} ({confidence*100:.2f}% confidence)")

            # GradCAM
            try:
                target_layer = model.layer4[1].conv2
                gradcam = GradCAM(model, target_layer)
                heatmap = gradcam.generate(input_tensor)

                img_np = np.array(img.resize((224, 224)))
                heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                superimposed_img = heatmap_color * 0.4 + img_np

                st.image(superimposed_img.astype(np.uint8), caption="Grad-CAM Heatmap", use_column_width=True)

            except Exception as e:
                st.warning(f"GradCAM generation failed: {e}")
