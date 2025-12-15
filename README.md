# Brain Tumor Detection from MRI Images  
### EfficientNet · Grad-CAM · Streamlit

This project detects **brain tumors from MRI images** using a deep learning model.  
It classifies an MRI image into two classes:

- **No Tumor**
- **Tumor**

To make the prediction understandable, the project includes **Explainable AI (XAI)** using **Grad-CAM**, which highlights the regions of the MRI image that influenced the model’s decision.  
A **Streamlit web application** is provided so users can upload MRI images and receive predictions interactively.

---

## Project Overview

This project performs the following tasks:

- Loads and preprocesses brain MRI images
- Trains a deep learning model using **transfer learning**
- Evaluates the model using standard classification metrics
- Visualizes model performance
- Explains model decisions using **Grad-CAM**
- Provides a user-friendly **Streamlit GUI**

---

## Dataset

The dataset used in this project is:

**Brain MRI Images for Brain Tumor Detection**  
Kaggle link:  
https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection

Expected dataset structure:

Class mapping:

- `no`  → 0  
- `yes` → 1  

---

## Model Architecture

- Base model: **EfficientNet-B0**
- Pretrained on: **ImageNet**
- Final classifier:
  - Dropout (0.4)
  - Fully connected layer with **1 output neuron**
- Task type: **Binary Classification**

---

## Training Details

- Image size: **224 × 224**
- Loss function: **BCEWithLogitsLoss**
- Optimizer: **Adam**
- Learning rate scheduler: **Cosine Annealing**
- Epochs: **20**
- Best model selected using **validation accuracy**

---

## Data Splitting

The dataset is split using **stratified sampling**:

- **Training set**: 70%
- **Validation set**: 15%
- **Test set**: 15%

This ensures balanced class distribution in all splits.

---

## Evaluation Metrics

Model performance is evaluated using:

- **Classification Report**
  - Precision
  - Recall
  - F1-score
- **Confusion Matrix**

---

## Explainable AI (XAI)

This project uses **Grad-CAM** to explain model predictions.

Two approaches are implemented:

1. **Custom Grad-CAM** (manual gradient and activation hooks)
2. **pytorch-grad-cam library**
   - GradCAM
   - GradCAM++
   - ScoreCAM

Grad-CAM heatmaps highlight important regions that influenced the **Tumor** prediction.

---

## Streamlit Web Application

The Streamlit application allows users to interact with the trained model.

### Home Page

- Project description
- Instructions
- Navigation to prediction page

### Prediction Page

- Upload MRI image (JPG / PNG)
- View:
  - **Positive (Tumor)** or **Negative (No Tumor)**
  - Prediction confidence
  - Class probabilities
  - **Grad-CAM visualization (only if Tumor is detected)**

User-adjustable settings:

- Tumor decision threshold
- Show / hide probabilities
- Show / hide Grad-CAM

---

## Model File Requirement

If the filename is different, update the `MODEL_PATH` variable in `app.py`.

---

## How to Run the Streamlit App

You can access the deployed Streamlit application here:

👉 **[Brain Tumor Detection Streamlit App](https://medical-image-analysis-for-brain-tumor-detection-using-mri.streamlit.app/)**




