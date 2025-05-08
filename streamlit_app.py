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
uploaded_files = None  # (optional, just for safety too)

# --- Sidebar for Navigation ---
st.sidebar.title("Navigation")

# Choose mode: Demo or Interactive
mode = st.sidebar.radio("Select Mode", ["Demo Mode (Guided Tour)", "Interactive Mode (Upload & Train)"])

# --- DEMO MODE: Guided Tour with Explanations and Next Button Navigation ---

# Initialize session state for demo step
if "demo_step" not in st.session_state:
    st.session_state.demo_step = 0

demo_sections = [
    "Pipeline Overview",
    "View Raw Data",
    "Tile Images",
    "Training Results",
    "GradCAM and SHAP Visualizations"
]

# Sidebar demo navigation (auto-updated by "Next" button)
demo_section = st.sidebar.radio(
    "Demo Steps",
    demo_sections,
    index=st.session_state.demo_step
)

# DEMO SECTION: Pipeline Overview
if demo_section == "Pipeline Overview":
    st.header("🔬 Pipeline Overview")
    st.write("""
        This project implements a deep learning pipeline for diagnosing lymphoid malignancies using histopathological slide images.
        Diagnosis of hematologic cancers, such as lymphoid malignancies, is subject to several challenges through manual slide review including variability and lack of personnel (Ivanova, 2024). 
        This project seeks to implement deep learning to automate and improve the accuracy of diagnosing lymphoid malignancies from whole-slide images
        The flowchart below summarizes the end-to-end process including data upload, tiling, preprocessing, model training, and explainability.
    """)
    st.image("demo_assets/pipeline_flowchart.png", caption="Project Pipeline")
    if st.button("➡ Next: View Raw Data"):
        st.session_state.demo_step += 1

# DEMO SECTION: Raw WSIs
elif demo_section == "View Raw Data":
    st.header("🧫 Sample Raw Whole Slide Images (WSIs)")
    st.write("""
        Whole Slide Images (WSIs) are high-resolution pathology scans of tissue samples.
        They are very large in size and must be tiled into smaller patches before training a neural network.
    """)
    st.image(["notebooks\demo_assets\sample_raw1.png"], width=300)
    if st.button("➡ Next: View Tiled Patches"):
        st.session_state.demo_step += 1

# DEMO SECTION: Tiled Image Patches
elif demo_section == "Tile Images":
    st.header("🧩 Tiled Patches")
    st.write("""
        WSIs are split into smaller 224x224 image tiles to allow for efficient training on GPUs and more localized pattern recognition.
        Below is a sample tile generated from a WSI.
    """)
    st.image(["notebooks/demo_assets/sample_tile1.png"], width=150)
    if st.button("➡ Next: Training Results"):
        st.session_state.demo_step += 1

# DEMO SECTION: Training Curve
elif demo_section == "Training Results":
    st.header("📈 Model Training Results")
    st.write("""
        The ResNet18 model is trained using tile patches. Accuracy and F1 score are evaluated on a validation set.
       
    """)
    st.image("notebooks/demo_assets/sample_processed_tile1.png", caption="Preprocessed Tile (224x224)")
    st.write("📊 **Accuracy**: 92% &nbsp;&nbsp;&nbsp;&nbsp; **F1 Score**: 0.90")
    if st.button("➡ Next: GradCAM + SHAP Explanations"):
        st.session_state.demo_step += 1

# DEMO SECTION: Explainability
elif demo_section == "GradCAM/SHAP Visualizations":
    st.header("GradCAM & Model Evaluation Visualizations")
    st.write("""
        To increase trust and transparency in model predictions, we use explainability tools:
        
        - **GradCAM** highlights image regions that influenced the model’s decision.
        - **SHAP** shows the global importance of features across the dataset.

        These visualizations can help researchers and clinicians interpret the model's behavior.
    """)
    st.image("notebooks/demo_assets/sample_gradcam_output.png", caption="GradCAM Heatmap")
   

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
            model = models.resnet18(pretrained=False)
            model.fc = nn.Linear(model.fc.in_features, 3)  # Adjust if more/less classes
            model.load_state_dict(torch.load("data/models/trained_resnet18.pth", map_location=device))
            model = model.to(device)
            model.eval()

            # Predict
            with torch.no_grad():
                outputs = model(input_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                top_prob, top_class = probs.topk(1, dim=1)

            class_labels = ["CLL", "FL", "MCL"]  # Replace with your class labels
            predicted_label = class_labels[top_class.item()]
            confidence = top_prob.item()

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
