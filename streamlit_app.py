# --- Imports ---
import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import cv2
import sys
from PIL import Image
import requests
from io import BytesIO
from fpdf import FPDF
sys.path.append("./scripts")
from gradcam_utils import GradCAM
from scripts.evaluate import predict_tile
import os
from datetime import datetime
from scripts.model_utils import load_model


# --- Tabs Navigation ---
st.set_page_config(page_title="Lymphoid Assistant", layout="wide", 
                   page_icon="🧬", initial_sidebar_state="expanded")

tab_demo, tab_interactive, tab_help = st.tabs(["🧪 Demo Mode", "🔍 Interactive Mode", "❓ Help"])

# --- Device setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Define Standard Transform ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

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

# --- INTERACTIVE MODE --- (Including Feedback and PDF Generation)
with tab_interactive:
    st.title("Interactive Inference & Visualization")

    uploaded_tile = st.file_uploader("Upload a tile image (.png, .jpg, .tif)", type=["png", "jpg", "tif"])
    url = st.text_input("Or paste image URL (optional)")
    img = None

    # --- Security Check ---
    if uploaded_tile is not None and uploaded_tile.size > 5_000_000:
        st.error("Image too large. Please upload a file under 5MB.")

    # Initialize Variables for Inference
    predicted_label = None
    confidence = None
    true_label = None
    report = None
    confusion = None

    # --- Load Image ---
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

    # --- Inference and GradCAM (Automatically triggered) ---
    if img:
        with st.spinner("🔎 Running inference and generating GradCAM..."):
            preprocess = transforms.Compose([ 
                transforms.Resize((224, 224)), 
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            model = load_model(device)

            class_labels = ["CLL", "FL", "MCL"]
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_label = class_labels[predicted_class.item()]
            confidence = confidence.item()

            st.success(f"🧠 **Prediction**: {predicted_label} ({confidence*100:.2f}% confidence)")

        # --- Descriptive Statistics ---
        img_np = np.array(img.resize((224, 224)))

        # Calculate descriptive statistics for each RGB channel
        mean_rgb = np.mean(img_np, axis=(0, 1))  # Mean for each channel (R, G, B)
        median_rgb = np.median(img_np, axis=(0, 1))  # Median for each channel
        std_rgb = np.std(img_np, axis=(0, 1))  # Standard deviation for each channel

        st.subheader("📊 Descriptive Statistics")
        st.write(f"**Mean (RGB)**: {mean_rgb}")
        st.write(f"**Median (RGB)**: {median_rgb}")
        st.write(f"**Standard Deviation (RGB)**: {std_rgb}")

        # --- Grad-CAM ---
        def apply_clahe(img: Image.Image) -> Image.Image:
            """Apply CLAHE to RGB image."""
            img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(img_cv)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            l = clahe.apply(l)
            img_clahe = cv2.merge((l, a, b))
            img_rgb = cv2.cvtColor(img_clahe, cv2.COLOR_LAB2RGB)
            return Image.fromarray(img_rgb)

        try:
            preprocess = transforms.Compose([ 
                transforms.Lambda(apply_clahe), 
                transforms.Resize((224, 224)),
                transforms.ToTensor(), 
                transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) 
            ])
            input_tensor = preprocess(img).unsqueeze(0).to(device)

            model = load_model()

            class_labels = ["CLL", "FL", "MCL"]
            output = model(input_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            predicted_label = class_labels[predicted_class.item()]
            confidence = confidence.item()

            # --- Grad-CAM ---
            target_layer = model.layer4[1].conv2
            gradcam = GradCAM(model, target_layer)
            heatmap = gradcam.generate(input_tensor)

            img_np = np.array(img.resize((224, 224)))
            heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
            heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap_resized), cv2.COLORMAP_JET)
            superimposed_img = heatmap_color * 0.4 + img_np

            gradcam_img_path = "gradcam_image.png"
            Image.fromarray(superimposed_img.astype(np.uint8)).save(gradcam_img_path)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.image(superimposed_img.astype(np.uint8), caption="Grad-CAM Heatmap", use_container_width=True)

            # --- RGB Histogram ---
            color_channels = ['r', 'g', 'b']
            fig_hist, ax_hist = plt.subplots()
            for i, color in enumerate(color_channels):
                hist = cv2.calcHist([img_np], [i], None, [256], [0, 256])
                ax_hist.plot(hist, color=color)
            ax_hist.set_title('RGB Histogram')
            with col2:
                st.pyplot(fig_hist)

            # --- Boxplot ---
            fig_box, ax_box = plt.subplots()
            ax_box.boxplot(
                [img_np[:, :, 0].flatten(), img_np[:, :, 1].flatten(), img_np[:, :, 2].flatten()],
                tick_labels=["Red", "Green", "Blue"]
            )
            ax_box.set_title('Pixel Intensity Distribution')
            with col3:
                st.pyplot(fig_box)

        except Exception as e:
            st.warning(f"Grad-CAM generation failed: {e}")

        # --- PDF Report Generation ---
        if img:
            # Generate report text without performance metrics
            report_text = f"""
            # Analysis Report
            
            ## Prediction
            **Prediction:** {predicted_label}  
            **Confidence:** {confidence:.2%}

            """

            # Initialize PDF
            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()

            # Add Title
            pdf.set_font("Arial", size=16, style='B')
            pdf.cell(200, 10, txt="Lymphoid Malignancy Classification Report", ln=True, align="C")
            pdf.ln(10)

            # Add Summary Text
            pdf.set_font("Arial", size=12)
            pdf.multi_cell(0, 10, report_text)

            # Add Image: Original Image
            pdf.ln(5)
            img_path = "uploaded_image.png"
            img.save(img_path)
            pdf.image(img_path, x=10, w=180)
            pdf.ln(10)

            # Add Image: Grad-CAM
            gradcam_img_path = "gradcam_image.png"
            Image.fromarray(superimposed_img.astype(np.uint8)).save(gradcam_img_path)
            pdf.image(gradcam_img_path, x=10, w=180)
            pdf.ln(10)

            # --- RGB Histogram ---
            img_tensor = transform(img)  # This is what's used for model input
            img_np = img_tensor.mul(255).byte().numpy().transpose(1, 2, 0)  # CHW → HWC for RGB

            # Safely split into RGB channels
            r_vals = img_np[:, :, 0].flatten()
            g_vals = img_np[:, :, 1].flatten()
            b_vals = img_np[:, :, 2].flatten()

            fig1, ax1 = plt.subplots()
            ax1.hist([r_vals, g_vals, b_vals], bins=256, color=['r', 'g', 'b'], label=['Red', 'Green', 'Blue'], alpha=0.6)
            ax1.set_title("RGB Color Histogram")
            ax1.set_xlabel("Pixel Intensity")
            ax1.set_ylabel("Frequency")
            ax1.legend()

            hist_img_path = "rgb_histogram.png"
            fig1.savefig(hist_img_path)
            plt.close(fig1)  # Close to free memory

            pdf.image(hist_img_path, x=10, w=180)
            pdf.ln(10)

            # --- Boxplot ---
            fig2, ax2 = plt.subplots()
            ax2.boxplot([r_vals, g_vals, b_vals], labels=['Red', 'Green', 'Blue'])
            ax2.set_title("Pixel Intensity Distribution")
            ax2.set_ylabel("Intensity")

            boxplot_img_path = "boxplot_rgb.png"
            fig2.savefig(boxplot_img_path)
            plt.close(fig2)

            pdf.image(boxplot_img_path, x=10, w=180)
            pdf.ln(10)

            # Save PDF to a buffer
            pdf_output = pdf.output(dest='S').encode('latin1')
            
            # Provide download link for the PDF report
            st.download_button("📄 Download PDF Report", data=pdf_output, file_name="analysis_report.pdf", key="pdf_report_download")

        # --- Feedback Section ---
        st.markdown("### Was this prediction correct?")

        # Use session state for feedback selection
        if "feedback_option" not in st.session_state:
            st.session_state.feedback_option = "Yes"

        st.radio("Select an option:", ["Yes", "No"], key="feedback_option")

        feedback_dir = "feedback_dir"
        os.makedirs(feedback_dir, exist_ok=True)

        # If feedback is No, show correction dropdown and store it
        if st.session_state.feedback_option == "No":
            if "corrected_label" not in st.session_state:
                st.session_state.corrected_label = "CLL"
            st.selectbox("What should the correct class be?", ["CLL", "FL", "MCL"], key="corrected_label")

        # Submit button with logic
        if st.button("Submit Error Report"):
            if st.session_state.feedback_option == "No" and "corrected_label" in st.session_state:
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                filename_base = f"{timestamp}_{predicted_label}_wrong-{st.session_state.corrected_label}"

                # Save image
                img_save_path = os.path.join(feedback_dir, filename_base + ".jpg")
                img.save(img_save_path)

                # Append to log
                log_path = os.path.join(feedback_dir, "feedback_log.csv")
                with open(log_path, "a") as f:
                    f.write(f"{filename_base},{predicted_label},{st.session_state.corrected_label},{confidence:.4f}\n")

                st.success("✅ Thank you! Your feedback and image were saved.")
            else:
                st.warning("Please select 'No' and choose the correct class before submitting.")

# --- HELP TAB ---
with tab_help:
    st.title("🛠 Help & Documentation")
    st.markdown(""" 
    **App Overview**  
    This app classifies lymphoma subtypes from histopathology tiles using deep learning. You can upload images, run classification, visualize results with GradCAM, and explore image features.
    
    **Link to documentation**
     https://github.com/Jacinda-G/-lymphoid-malignancy-classification
                 
    **User Guide (click view raw to download)**
    https://github.com/Jacinda-G/-lymphoid-malignancy-classification/blob/main/Lymphoid%20Malignancy%20Product%20User%20Guide.docx
                 
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
