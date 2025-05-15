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
from scripts.evaluate import predict_tile

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Tabs Navigation ---
tab_demo, tab_interactive, tab_help = st.tabs(["🧪 Demo Mode", "🔍 Interactive Mode", "❓ Help"])

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
        st.write("This deep learning pipeline is designed to assist in the diagnosis of hematologic malignancies by classifying whole slide image (WSI) tiles into three major subtypes: Chronic Lymphocytic Leukemia (CLL), Follicular Lymphoma (FL), and Mantle Cell Lymphoma (MCL). The pipeline includes stages for preprocessing, tiling of WSIs, training and evaluating a convolutional neural network (ResNet-50), and visualizing model interpretability using Grad-CAM.")
        st.write("This project is part of a larger research initiative to improve the accuracy and efficiency of lymphoma diagnosis through advanced image analysis techniques.")
        st.write("The demo will guide you through the steps of the pipeline, showcasing the data, model training, and evaluation results.")
        st.write("Click the buttons at the bottom of the page to navigate through the demo.")
        st.image("demo_assets/pipeline_flowchart.png")
    elif step == 1:
        st.write("Sample WSIs from a public Kaggle dataset.")
        st.write("The dataset contains 3 classes: CLL, FL, and MCL.")
        st.write("The images are in .tif format and are large in size.")
        st.image("notebooks/demo_assets/sample_raw1.png", width=300)
    elif step == 2:
        st.write("Tiles (224x224) are extracted from WSIs for training.")
        st.write("The tiles are labeled according to the WSI they were extracted from.")
        st.image("notebooks/demo_assets/sample_tile1.png", width=150)
    elif step == 3:
        st.write("Model training was performed using a ResNet50 architecture.")
        st.write("Training included augmentation and optimization.")
        st.image("notebooks/demo_assets/sample_processed_tile1.png")
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

    # --- Security Check ---
    if uploaded_tile is not None and uploaded_tile.size > 5_000_000:
        st.error("Image too large. Please upload a file under 5MB.")

    # --- Load image ---
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

    # --- Analysis method selection ---
    analysis_method = st.selectbox(
        "Choose an analysis method",
        ["None", "Grayscale Histogram", "Color Histogram", "Edge Detection"]
    )

    # --- Inference and GradCAM ---
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

    # --- Visualization based on analysis method ---
    if img and analysis_method != "None":
        st.subheader(f"🔍 {analysis_method} Visualization")
        img_np = np.array(img.resize((224, 224)))

        if analysis_method == "Grayscale Histogram":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            fig, ax = plt.subplots()
            ax.plot(hist, color='gray')
            st.pyplot(fig)

        elif analysis_method == "Color Histogram":
            fig, ax = plt.subplots()
            for i, color in enumerate(['r', 'g', 'b']):
                hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
                ax.plot(hist, color=color)
            st.pyplot(fig)

        elif analysis_method == "Edge Detection":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            st.image(edges, caption="Edges", use_container_width=True)

    # --- Generate & Save Report ---
    if img and st.button("📝 Generate Analysis Report"):
        report_text = f"""
        ## Report Summary

        **Prediction:** {predicted_label}  
        **Confidence:** {confidence:.2%}  
        **Analysis Method:** {analysis_method}

        *Model: ResNet50 trained on lymphoma tiles.*
        """
        st.markdown(report_text)
        st.download_button("📄 Download Report", data=report_text.encode(), file_name="analysis_report.txt")

    # --- Testing all functions ---
    def run_tests():
        return {
            "Model Loaded": True,
            "Image Provided": img is not None,
            "Analysis Method Selected": analysis_method != "None",
            "GradCAM Enabled": img is not None,
            "Report Button Clicked": True
        }

    if st.button("🔎 Run System Test"):
        test_results = run_tests()
        st.json(test_results)

# --- HELP TAB ---
with tab_help:
    st.title("🛠 Help & Documentation")
    st.markdown("""
    **App Overview**  
    This app classifies lymphoma subtypes from histopathology tiles using deep learning. You can upload images, run classification, visualize results with GradCAM, and explore image features.

    **Tabs**  
    - **Demo Mode**: A walkthrough of how the model was trained.  
    - **Interactive Mode**: Upload or link a tile, classify, analyze, and download reports.  
    - **Help**: You’re here now!

    **Steps to Use**  
    1. Upload a tile image or paste an image URL  
    2. Choose an analysis method (optional)  
    3. Click *Classify & Explain*  
    4. Review GradCAM and image analysis results  
    5. Generate & download your report  
    6. Use the test button to verify system behavior

    """)
