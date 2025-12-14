# app.py
# Streamlit app: Landing page (Next) -> Upload/Result page with Positive/Negative + Grad-CAM XAI (only if Positive)

import os
import base64
from typing import Dict, Optional

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# XAI (Grad-CAM)
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import BinaryClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image


# =========================
# App Configuration
# =========================
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths (expected in your GitHub repo)
MODEL_PATH = os.path.join( "efficientnet_brain_tumor_best.pth")
BG_PATH = os.path.join("background.jpg")

IMG_SIZE = 224
CLASS_NAMES = ["No Tumor", "Tumor"]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================
# Styling Helpers
# =========================
def set_bg(image_path: str) -> None:
    """Sets a full-page background image with a dark overlay. Safe if image missing."""
    if not os.path.exists(image_path):
        return
    with open(image_path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
              linear-gradient(rgba(0,0,0,0.60), rgba(0,0,0,0.60)),
              url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_home_card_css() -> None:
    st.markdown(
        """
        <style>
        .glass-card {
            background: rgba(255,255,255,0.10);
            border: 1px solid rgba(255,255,255,0.18);
            box-shadow: 0 10px 35px rgba(0,0,0,0.35);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border-radius: 18px;
            padding: 28px;
            color: white;
        }
        .title {
            font-size: 44px;
            font-weight: 750;
            margin: 0 0 8px 0;
        }
        .subtitle {
            font-size: 18px;
            line-height: 1.65;
            opacity: 0.95;
        }
        .small-note {
            font-size: 13px;
            opacity: 0.85;
            line-height: 1.55;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_predict_bg() -> None:
    """Simple background for prediction page."""
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Model + Preprocessing
# =========================
transform = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
    ]
)


@st.cache_resource
def load_model() -> nn.Module:
    """
    Loads EfficientNet-B0 with the same classifier head used in training.
    Expects a state_dict saved via: torch.save(model.state_dict(), MODEL_PATH)
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Ensure it exists in your repo."
        )

    model = models.efficientnet_b0(weights=None)

    # Binary classifier head (matches your notebook)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.4, inplace=True),
        nn.Linear(in_features, 1),
    )

    state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    return model


def predict_probs(pil_image: Image.Image) -> Dict[str, float]:
    """
    Returns class probabilities: {"No Tumor": p0, "Tumor": p1}
    """
    model = load_model()

    img = pil_image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x).view(-1)
        p_tumor = torch.sigmoid(logits).item()

    return {"No Tumor": float(1.0 - p_tumor), "Tumor": float(p_tumor)}


def tensor_to_rgb_float(image_tensor: torch.Tensor) -> np.ndarray:
    """
    Converts a normalized tensor (3,H,W) -> RGB float image (H,W,3) in [0,1]
    """
    x = image_tensor.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    x = (x * IMAGENET_STD) + IMAGENET_MEAN
    x = np.clip(x, 0.0, 1.0)
    return x


def get_target_layer(model: nn.Module) -> nn.Module:
    """
    Attempts to select an appropriate target layer for Grad-CAM on EfficientNet.
    Tries a couple of common choices for robustness.
    """
    # Common choice
    try:
        return model.features[-1]
    except Exception:
        pass

    # Alternative choice some notebooks use
    try:
        return model.features[-1][0]
    except Exception:
        pass

    # Last resort: any module inside features
    for m in reversed(list(model.features.modules())):
        # choose a non-container leaf module
        if len(list(m.children())) == 0:
            return m

    raise RuntimeError("Could not determine a target layer for Grad-CAM.")


def compute_gradcam_overlay(pil_image: Image.Image) -> np.ndarray:
    """
    Computes Grad-CAM overlay image (RGB uint8) for the positive class "Tumor".
    """
    model = load_model()

    img = pil_image.convert("RGB")
    x = transform(img)  # (3,H,W) normalized
    input_tensor = x.unsqueeze(0).to(DEVICE)

    target_layer = get_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])

    # Positive class target (Tumor = 1)
    targets = [BinaryClassifierOutputTarget(1)]

    # grayscale_cam shape: (H,W)
    grayscale_cam = cam(input_tensor=input_tensor, targets=targets)[0]

    rgb_img = tensor_to_rgb_float(x)  # float in [0,1]
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)  # uint8 RGB
    return overlay


# =========================
# Navigation State
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"  # "home" or "predict"


# =========================
# Pages
# =========================
def render_home() -> None:
    set_bg(BG_PATH)
    inject_home_card_css()

    col_left, col_right = st.columns([1.25, 1])

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="title">Brain Tumor Detection</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="subtitle">
            This application demonstrates a deep-learning model (EfficientNet-based) that classifies brain MRI images into
            <b>Tumor</b> vs <b>No Tumor</b>.
            <br><br>
            <b>Workflow</b>
            <ol>
              <li>Click <b>Next</b></li>
              <li>Upload an MRI image (JPG/PNG)</li>
              <li>Press <b>Result</b> to view prediction</li>
              <li>If the result is <b>Positive</b>, the app will show an <b>Explainable AI</b> heatmap (Grad-CAM)</li>
            </ol>
            </div>
            <div class="small-note">
            Disclaimer: This is a research/educational demonstration and is not a medical device.
            Do not use it for clinical diagnosis or treatment decisions.
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.write("")
        if st.button("Next →", type="primary"):
            st.session_state.page = "predict"
            st.rerun()

        st.markdown("</div>", unsafe_allow_html=True)

    with col_right:
        st.markdown(
            """
            <div class="glass-card">
              <h3 style="margin-top:0;">What you will see</h3>
              <ul style="line-height:1.8;">
                <li><b>Positive / Negative</b> result</li>
                <li>Class probabilities</li>
                <li><b>Grad-CAM</b> explanation for Positive results</li>
              </ul>
              <div class="small-note">
                For best results, use clear MRI slices. The model output is probabilistic, not definitive.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_predict() -> None:
    set_predict_bg()

    top = st.columns([1, 2, 1])
    with top[0]:
        if st.button("← Back"):
            st.session_state.page = "home"
            st.rerun()

    st.title("Upload MRI and Get Result")

    # Safety / UX: show model availability early
    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found at: `{MODEL_PATH}`. "
            f"Upload your weights to GitHub at that path (or update MODEL_PATH)."
        )
        st.stop()

    with st.expander("Settings", expanded=False):
        threshold = st.slider("Decision threshold for Tumor (Positive)", 0.10, 0.90, 0.50, 0.01)
        show_probs = st.checkbox("Show probabilities", value=True)
        show_xai = st.checkbox("Show XAI (Grad-CAM) when Positive", value=True)

    uploaded = st.file_uploader("Upload a brain MRI image (JPG/PNG)", type=["jpg", "jpeg", "png"])

    if uploaded is None:
        st.info("Upload an image to enable prediction.")
        return

    img = Image.open(uploaded)
    st.image(img, caption="Uploaded MRI Image", use_column_width=True)

    if st.button("Result", type="primary"):
        with st.spinner("Running inference..."):
            probs = predict_probs(img)

        prob_tumor = probs["Tumor"]
        is_positive = prob_tumor >= threshold

        # Decision label + confidence
        decision = "Positive (Tumor)" if is_positive else "Negative (No Tumor)"
        confidence = (prob_tumor if is_positive else probs["No Tumor"]) * 100.0

        st.subheader("Result")
        if is_positive:
            st.error(f"Prediction: **{decision}**  |  Confidence: **{confidence:.2f}%**")
        else:
            st.success(f"Prediction: **{decision}**  |  Confidence: **{confidence:.2f}%**")

        if show_probs:
            st.subheader("Probabilities")
            st.json(probs)

        # XAI only if Positive
        if is_positive and show_xai:
            st.subheader("Explainable AI (E-AI): Grad-CAM")
            st.write(
                "The heatmap highlights regions that contributed most to the model’s **Tumor** decision. "
                "Brighter areas indicate stronger influence."
            )
            with st.spinner("Generating explanation (Grad-CAM)..."):
                try:
                    overlay = compute_gradcam_overlay(img)
                    st.image(
                        overlay,
                        caption="Grad-CAM Overlay (Tumor Evidence Regions)",
                        use_column_width=True,
                    )
                except Exception as e:
                    st.warning(
                        "Grad-CAM could not be generated (layer mismatch or dependency issue). "
                        "Your prediction is still shown above."
                    )
                    st.code(str(e))


# =========================
# Router
# =========================
if st.session_state.page == "home":
    render_home()
else:
    render_predict()
