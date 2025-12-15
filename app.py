import os
import base64
from typing import Dict

import streamlit as st
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import matplotlib.cm as cm


# =========================
# App Configuration
# =========================
st.set_page_config(page_title="Brain Tumor Detection", layout="wide")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = os.path.join("efficientnet_brain_tumor_best.pth")
BG_PATH = os.path.join("background.jpg")

IMG_SIZE = 224

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# =========================
# UI Styling Helpers
# =========================
def set_bg(image_path: str) -> None:
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
              linear-gradient(rgba(0,0,0,0.65), rgba(0,0,0,0.65)),
              url("data:image/jpg;base64,{b64}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def inject_global_css() -> None:
    st.markdown(
        """
        <style>
        /* Glass card + improved readability */
        .glass-card {
            background: rgba(10, 18, 32, 0.72);
            border: 1px solid rgba(255,255,255,0.16);
            box-shadow: 0 12px 40px rgba(0,0,0,0.45);
            backdrop-filter: blur(10px);
            -webkit-backdrop-filter: blur(10px);
            border-radius: 18px;
            padding: 28px;
            color: #F8FAFC;
        }
        .title {
            font-size: 44px;
            font-weight: 800;
            margin: 0 0 8px 0;
            color: #F8FAFC;
            text-shadow: 0 2px 12px rgba(0,0,0,0.55);
        }
        .subtitle {
            font-size: 18px;
            line-height: 1.7;
            color: rgba(248,250,252,0.92);
        }
        .small-note {
            font-size: 13px;
            color: rgba(248,250,252,0.80);
            line-height: 1.6;
        }

        /* Probability cards */
        .prob-wrap {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 14px;
            margin-top: 8px;
        }
        .prob-card {
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.14);
            border-radius: 14px;
            padding: 14px 14px 12px 14px;
        }
        .prob-label {
            font-size: 14px;
            color: rgba(248,250,252,0.85);
            margin-bottom: 6px;
        }
        .prob-value {
            font-size: 26px;
            font-weight: 800;
            color: #F8FAFC;
            margin-bottom: 10px;
        }
        .prob-bar-bg {
            height: 10px;
            background: rgba(255,255,255,0.10);
            border-radius: 999px;
            overflow: hidden;
        }
        .prob-bar-fill {
            height: 100%;
            width: 0%;
            background: rgba(56,189,248,0.95);
            border-radius: 999px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def set_predict_bg() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #0b1220; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =========================
# Preprocessing
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
    EfficientNet-B0 with binary head, loads state_dict.
    Must match your training architecture.
    """
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at '{MODEL_PATH}'. Upload it to GitHub at that path."
        )

    model = models.efficientnet_b0(weights=None)

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
    model = load_model()
    img = pil_image.convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(x).view(-1)
        p_tumor = torch.sigmoid(logits).item()

    return {"No Tumor": float(1.0 - p_tumor), "Tumor": float(p_tumor)}


def render_probabilities_card(probs: Dict[str, float]) -> None:
    no_tumor = float(np.clip(probs.get("No Tumor", 0.0), 0.0, 1.0))
    tumor = float(np.clip(probs.get("Tumor", 0.0), 0.0, 1.0))

    html = f"""
    <div class="prob-wrap">
      <div class="prob-card">
        <div class="prob-label">No Tumor</div>
        <div class="prob-value">{no_tumor*100:.2f}%</div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{no_tumor*100:.2f}%"></div></div>
      </div>

      <div class="prob-card">
        <div class="prob-label">Tumor</div>
        <div class="prob-value">{tumor*100:.2f}%</div>
        <div class="prob-bar-bg"><div class="prob-bar-fill" style="width:{tumor*100:.2f}%"></div></div>
      </div>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)


# =========================
# Manual Grad-CAM (NO OpenCV)
# =========================
def _find_target_layer(model: nn.Module) -> nn.Module:
    """
    Select a reasonable last convolutional-ish layer for EfficientNet.
    We target model.features[-1] by default.
    """
    try:
        return model.features[-1]
    except Exception:
        for m in reversed(list(model.features.modules())):
            if len(list(m.children())) == 0:
                return m
    raise RuntimeError("Could not locate a target layer for Grad-CAM.")


def _tensor_to_rgb_float(x_norm: torch.Tensor) -> np.ndarray:
    """
    x_norm: normalized tensor (3,H,W)
    Returns RGB float image (H,W,3) in [0,1]
    """
    x = x_norm.detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)
    x = (x * IMAGENET_STD) + IMAGENET_MEAN
    x = np.clip(x, 0.0, 1.0)
    return x


def _resize_cam(cam: np.ndarray, out_h: int, out_w: int) -> np.ndarray:
    """
    Resize CAM using PIL (no cv2).
    cam: (h,w) float
    """
    cam_img = Image.fromarray((cam * 255).astype(np.uint8))
    cam_img = cam_img.resize((out_w, out_h), resample=Image.BILINEAR)
    cam_resized = np.asarray(cam_img).astype(np.float32) / 255.0
    return cam_resized


def compute_gradcam_overlay(pil_image: Image.Image) -> np.ndarray:
    """
    Returns an overlay image (H,W,3) uint8 with heatmap blended into the MRI.
    """
    model = load_model()
    target_layer = _find_target_layer(model)

    activations = {}
    gradients = {}

    def fwd_hook(module, inp, out):
        activations["value"] = out

    def bwd_hook(module, grad_in, grad_out):
        gradients["value"] = grad_out[0]

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    try:
        img = pil_image.convert("RGB")
        x_norm = transform(img)  # (3,H,W)
        input_tensor = x_norm.unsqueeze(0).to(DEVICE)

        logits = model(input_tensor).view(-1)  # shape (1,)
        score = logits[0]  # encourage Tumor (positive)

        model.zero_grad(set_to_none=True)
        score.backward(retain_graph=False)

        A = activations["value"]          # (N,C,h,w)
        dA = gradients["value"]           # (N,C,h,w)

        weights = torch.mean(dA, dim=(2, 3), keepdim=True)  # (N,C,1,1)
        cam = torch.sum(weights * A, dim=1)  # (N,h,w)
        cam = torch.relu(cam)[0]             # (h,w)

        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam_np = cam.detach().cpu().numpy().astype(np.float32)

        rgb_img = _tensor_to_rgb_float(x_norm)
        H, W = rgb_img.shape[:2]
        cam_resized = _resize_cam(cam_np, H, W)

        heatmap = cm.get_cmap("jet")(cam_resized)[..., :3]  # (H,W,3) float [0,1]

        alpha = 0.45
        overlay = (1 - alpha) * rgb_img + alpha * heatmap
        overlay = np.clip(overlay, 0.0, 1.0)
        overlay_uint8 = (overlay * 255).astype(np.uint8)

        return overlay_uint8

    finally:
        h1.remove()
        h2.remove()


# =========================
# Navigation
# =========================
if "page" not in st.session_state:
    st.session_state.page = "home"


def render_home() -> None:
    set_bg(BG_PATH)
    inject_global_css()

    col_left, col_right = st.columns([1.25, 1])

    with col_left:
        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown('<div class="title">Brain Tumor Detection</div>', unsafe_allow_html=True)
        st.markdown(
            """
            <div class="subtitle">
            This web application demonstrates an EfficientNet-based deep learning model that classifies brain MRI images
            as <b>Tumor</b> or <b>No Tumor</b>.
            <br><br>
            <b>Steps</b>
            <ol>
              <li>Click <b>Next</b></li>
              <li>Upload an MRI image (JPG/PNG)</li>
              <li>Press <b>Result</b> to view prediction</li>
              <li>If <b>Positive</b>, an Explainable AI heatmap (Grad-CAM) will be shown</li>
            </ol>
            </div>
            <div class="small-note">
            Disclaimer: For educational/research purposes only. Not for medical diagnosis.
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
              <h3 style="margin-top:0; color:#F8FAFC;">Outputs</h3>
              <ul style="line-height:1.8; color:rgba(248,250,252,0.92);">
                <li>Clear <b>Positive/Negative</b> decision</li>
                <li>Class probabilities</li>
                <li><b>Grad-CAM</b> overlay for Positive results</li>
              </ul>
              <div class="small-note">
                The heatmap indicates model attention, not a clinical explanation.
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_predict() -> None:
    set_predict_bg()
    inject_global_css()

    top = st.columns([1, 2, 1])
    with top[0]:
        if st.button("← Back"):
            st.session_state.page = "home"
            st.rerun()

    st.title("Upload MRI and Get Result")

    if not os.path.exists(MODEL_PATH):
        st.error(
            f"Model file not found: `{MODEL_PATH}`. Upload it to GitHub or update MODEL_PATH in app.py."
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

        p_tumor = probs["Tumor"]
        is_positive = p_tumor >= threshold

        decision = "Positive (Tumor)" if is_positive else "Negative (No Tumor)"
        confidence = (p_tumor if is_positive else probs["No Tumor"]) * 100.0

        st.subheader("Result")
        if is_positive:
            st.error(f"Prediction: **{decision}**  |  Confidence: **{confidence:.2f}%**")
        else:
            st.success(f"Prediction: **{decision}**  |  Confidence: **{confidence:.2f}%**")

        if show_probs:
            st.subheader("Probabilities")
            render_probabilities_card(probs)

        if is_positive and show_xai:
            st.subheader("Explainable AI: Grad-CAM")
            st.write(
                "The heatmap highlights regions that contributed most to the **Tumor** decision. "
                "Brighter regions indicate stronger influence."
            )
            with st.spinner("Generating explanation..."):
                try:
                    overlay = compute_gradcam_overlay(img)
                    st.image(overlay, caption="Grad-CAM Overlay (Tumor Evidence Regions)", use_column_width=True)
                except Exception as e:
                    st.warning("Could not generate Grad-CAM. Prediction is still valid.")
                    st.code(str(e))


if st.session_state.page == "home":
    render_home()
else:
    render_predict()
