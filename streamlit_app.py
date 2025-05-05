# --- Imports ---
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import cv2
import io
import sys
from PIL import Image
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

# --- Interactive Mode ---
elif mode == "Interactive Mode (Upload & Train)":
    interactive_section = st.sidebar.radio(
        "Interactive Actions", 
        ["Upload Data", "Preprocess Data", "Train Model", "Inference & Visualize"]
    )

    uploaded_tile = None  # Default for inference page
    uploaded_files = None

    # --- Upload Data Section ---
    if interactive_section == "Upload Data":
        st.header("Upload Your Own Data")
        st.write("Upload Whole Slide Images (WSIs) or tile patches to use in preprocessing or training a new model.")

        uploaded_files = st.file_uploader(
            "Upload WSIs or tile patches",
            type=["tif", "png", "jpg"],
            accept_multiple_files=True,
            help="Accepted formats: .tif, .png, .jpg. You can select multiple files."
        )
        
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files successfully!")
            st.image([Image.open(file) for file in uploaded_files], width=150)

    # --- Preprocess Data Section ---
    if interactive_section == "Preprocess Data":
        st.header("Data Preprocessing")
        st.write("You can normalize images, filter out poor quality tiles, and optionally retile images with overlap.")

        normalize = st.checkbox("Normalize Images", help="Adjusts intensity values to a standard range.")
        filter_tiles = st.checkbox("Filter Uninformative Tiles", help="Removes tiles that are too blank or noisy.")
        retile = st.checkbox("Tile WSIs with Overlap", help="Generate overlapping tiles for more data.")

        if st.button("Run Preprocessing"):
            st.success("✅ Preprocessing complete! (Note: This is a demo — real functionality coming soon.)")

    # --- Train Model Section ---
    if interactive_section == "Train Model":
        st.header("Train a ResNet Model")
        st.write("Use the uploaded tiles to train a deep learning model. This may take a few minutes depending on dataset size.")

        if st.button("Train ResNet18 Model"):
            st.success("✅ Training complete! (Note: This is a demo — real training runs in notebook!)")

    # --- Inference & Visualize Section ---
    if interactive_section == "Inference & Visualize":
        st.header("Upload and Classify a Tile Image")
        st.write("Upload a single tile to classify it into one of the lymphoma subtypes. You will also see a GradCAM heatmap.")

        uploaded_tile = st.file_uploader(
            "Choose a tile image (.png, .jpg, .tif)",
            type=["png", "jpg", "tif"],
            help="Upload one tile image for classification and visualization."
        )

        if uploaded_tile is not None:
            img = Image.open(uploaded_tile).convert("RGB")
            st.image(img, caption="Uploaded Image", use_column_width=True)

            if st.button("Classify Image"):
                # Preprocessing
                preprocess = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                         std=[0.5, 0.5, 0.5])
                ])
                input_tensor = preprocess(img).unsqueeze(0).to(device)

                # Load model
                model = models.resnet18(pretrained=False)
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, len(class_names))
                model.load_state_dict(torch.load("data/models/trained_resnet18.pth", map_location=device))
                model = model.to(device)
                model.eval()

                # Prediction
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    top_prob, top_class = probs.topk(1, dim=1)

                class_labels = list(class_names.keys())
                predicted_label = class_labels[top_class.item()]
                confidence = top_prob.item()

                st.success(f"Prediction: **{predicted_label}** with {confidence*100:.2f}% confidence.")

                # GradCAM (Optional)
                try:
                    target_layer = model.layer4[1].conv2
                    gradcam = GradCAM(model, target_layer)
                    heatmap = gradcam.generate(input_tensor)

                    # Overlay heatmap
                    img_array = np.array(img.resize((224, 224)))
                    heatmap_resized = cv2.resize(heatmap, (img_array.shape[1], img_array.shape[0]))
                    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
                    superimposed_img = heatmap_color * 0.4 + img_array

                    st.image(superimposed_img.astype(np.uint8), caption="Grad-CAM Heatmap", use_column_width=True)

                except Exception as e:
                    st.error("GradCAM visualization failed. Please check model compatibility.")
