# Lymphoid Malignancy Analysis Platform

## 📋 Project Overview
This project develops a deep learning-based platform to classify lymphoid malignancies using whole slide images (WSIs) and tile patch images.  
Users can upload histopathology images, preprocess them, train a ResNet-based model, visualize model interpretability with Grad-CAM and SHAP, and download analysis reports.

Built using:
- Python (PyTorch, Scikit-learn, OpenSlide)
- Streamlit (for web app)
- Matplotlib and SHAP for explainable AI

---

## 🚀 Features

- **Upload Data**: Upload WSIs or tile patches through the web interface.
- **Data Preprocessing**: Normalize images, remove uninformative tiles, retile WSIs with overlap.
- **Model Training**: Train a ResNet-18 CNN on uploaded or preprocessed tiles.
- **Inference**: Run predictions on unseen tiles.
- **Visualization**:
  - Grad-CAM attention heatmaps
  - Global SHAP feature attribution heatmaps
- **Reporting**: Generate and download classification reports (Accuracy, F1, Precision, Recall, ROC-AUC).

---

## 📂 Project Structure

```plaintext
lymphoid_malignancy_platform/
├── data/               # Local storage for uploaded tiles and intermediate data
│   ├── (empty, user uploads populate this folder)
│   ├── README.md        # Note: No raw WSI data in repo
├── notebooks/           # Exploratory notebooks
│   ├── tile_wsis.ipynb
│   ├── preprocess_tiles.ipynb
│   ├── load_tiles.ipynb
│   ├── train_model.ipynb
│   ├── evaluate_model.ipynb
│   ├── visualization.ipynb
├── scripts/             # Core functionality
│   ├── tiling.py
│   ├── preprocessing.py
│   ├── data_loader.py
│   ├── train_utils.py
├── streamlit_app.py     # Streamlit Web App
├── requirements.txt     # Required Python packages
├── README.md            # This file
