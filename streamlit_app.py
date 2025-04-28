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

# Navigation inside each mode
if mode == "Demo Mode (Guided Tour)":
    demo_section = st.sidebar.radio("Demo Steps", ["Pipeline Overview", "View Raw Data", "Tile Images", "Training Results", "GradCAM/SHAP Visualizations"])

    if demo_section == "Pipeline Overview":
        st.title("Pipeline Overview")
        st.write("""
            Welcome to the guided tour! This demo will walk you through:
            1. Viewing raw WSIs
            2. Generating tiles
            3. Training a model
            4. Visualizing model explanations (GradCAM and SHAP)
            """)
        st.header("Pipeline Overview")
        st.image("demo_assets/pipeline_flowchart.png", caption="Data Science Pipeline Overview")

    if demo_section == "View Raw Data":
        st.header("Sample Raw Data")
        st.image(["demo_assets/sample_raw1.png"], width=300)

    if demo_section == "Tile Images":
        st.header("Generated Tiles")
        st.image(["demo_assets/sample_tile1.png"], width=150)

    if demo_section == "Training Results":
        st.header("Training Results on Sample Tiles")
        st.image("demo_assets/sample_training_curve.png")
        st.write("Accuracy: 92%, F1 Score: 0.90")

    if demo_section == "GradCAM/SHAP Visualizations":
        st.header("GradCAM and SHAP on Sample Images")
        st.image("demo_assets/sample_gradcam_output.png", caption="GradCAM Heatmap")
        st.image("demo_assets/sample_shap_summary.png", caption="Global SHAP Summary")

# --- Interactive Mode ---
elif mode == "Interactive Mode (Upload & Train)":
    interactive_section = st.sidebar.radio("Interactive Actions", ["Upload Data", "Preprocess Data", "Train Model", "Inference & Visualize"])

    if interactive_section == "Upload Data":
        st.header("Upload Your Own Data")
        uploaded_files = st.file_uploader("Upload WSIs or tile patches", type=["tif", "png", "jpg"], accept_multiple_files=True)
        if uploaded_files:
            st.success(f"Uploaded {len(uploaded_files)} files successfully!")
            st.image([Image.open(file) for file in uploaded_files], width=150)

    if interactive_section == "Preprocess Data":
        st.header("Data Preprocessing")
        normalize = st.checkbox("Normalize Images")
        filter_tiles = st.checkbox("Filter Uninformative Tiles")
        retile = st.checkbox("Tile WSIs with Overlap")
        if st.button("Run Preprocessing"):
            st.success("Preprocessing complete!")

    if interactive_section == "Train Model":
        st.header("Train a ResNet Model")
        if st.button("Train ResNet18"):
            st.success("Model training complete!")

    if interactive_section == "Inference & Visualize":
        st.header("Classify a New Tile")
        uploaded_tile = st.file_uploader("Upload a single tile image for classification", type=["png", "jpg", "tif"])
        if uploaded_tile is not None:
            # your full inference + GradCAM code here
            st.success("Prediction complete!")
            # Show GradCAM heatmap
            # Show download button
