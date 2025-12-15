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

