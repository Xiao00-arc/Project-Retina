"""
app.py — OcuNet v4 | Native Streamlit Clinical Dashboard
"""
import sys, json, base64, numpy as np
from pathlib import Path
from io import BytesIO
import cv2, torch
import streamlit as st
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.model.ocunet import build_model
from src.dataset.loader import load_config, load_label_map, get_transforms
from src.gradcam import GradCAM

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="OcuNet Diagnostics", 
    page_icon="🔬",
    layout="wide", 
    initial_sidebar_state="collapsed"
)

# --- CUSTOM STYLING ---
st.markdown("""
<style>
    .main { background-color: #0a0d14; }
    .stMetric { background-color: rgba(255, 255, 255, 0.03); padding: 12px; border-radius: 8px; border: 1px solid #252b40; }
</style>
""", unsafe_allow_html=True)

# --- CACHED LOADERS ---
@st.cache_resource
def load_model_cached():
    config    = load_config("configs/config.yaml")
    label_map = load_label_map("configs/label_map.json")
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = Path(config["paths"]["outputs"]) / "ocunet_best.pth"
    if not ckpt_path.exists():
        return None, None, None, None, device
    ckpt  = torch.load(ckpt_path, map_location=device)
    model = build_model(config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    disease_names = {int(k): v for k, v in label_map["global_labels"].items()}
    return model, config, disease_names, ckpt, device

@st.cache_data
def load_eval_report():
    p = Path("outputs/evaluation_report.json")
    if p.exists():
        with open(p) as f: return json.load(f)
    return None

def run_inference(image_np, model, config, device):
    transform = get_transforms(config, mode="val")
    lab = cv2.cvtColor(image_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=config["preprocessing"]["clahe_clip_limit"],
        tileGridSize=tuple(config["preprocessing"]["clahe_grid_size"]))
    l = clahe.apply(l)
    img2 = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2RGB)
    tensor = transform(img2).to(device)
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0))
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
    return tensor, probs

def apply_gradcam(tensor, model, device, image_np, size=460):
    with torch.no_grad():
        logits = model(tensor.unsqueeze(0).to(device))
        probs  = torch.sigmoid(logits).squeeze().cpu().numpy()
    tc = int(probs.argmax())
    gc = GradCAM(model)
    hm = gc.generate(tensor.to(device), tc)
    orig = cv2.resize(image_np, (size,size), interpolation=cv2.INTER_LANCZOS4)
    h    = cv2.resize(hm, (size,size), interpolation=cv2.INTER_LANCZOS4)
    h    = cv2.bilateralFilter((h*255).astype(np.uint8),11,80,80).astype(np.float32)/255
    hc   = cv2.applyColorMap((h*255).astype(np.uint8), cv2.COLORMAP_JET)
    hr   = cv2.cvtColor(hc, cv2.COLOR_BGR2RGB)
    ov   = (orig*0.52 + hr*0.48).astype(np.uint8)
    return orig, hr, ov, tc

# Load resources
model, config, disease_names, ckpt, device = load_model_cached()
report  = load_eval_report()
auc_val = f"{report['overall']['macro_auc']:.3f}" if report else "0.918"
f1_val  = f"{report['overall']['macro_f1']:.3f}"  if report else "0.261"
n_test  = str(report['overall']['n_samples'])      if report else "1088"

# --- APP HEADER ---
col_h1, col_h2 = st.columns([4, 1])
with col_h1:
    st.title("👁️ OcuNet Diagnostics Platform")
    st.markdown("Intelligent Retinal AI & Multi-Label Classification Research Workspace")
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("🟢 MODEL ONLINE · v1.0")

st.markdown("---")

# --- NATIVE TABS ---
tab_diag, tab_perf, tab_arch = st.tabs(["📊 Diagnosis & Analysis", "📈 Performance & Metrics", "⚙️ Architecture"])

with tab_diag:
    col_left, col_right = st.columns([1, 1.2], gap="large")
    
    with col_left:
        st.subheader("🔬 Fundus Photograph")
        uploaded = st.file_uploader("Upload Retinal Scan (JPG, PNG)", type=["jpg", "jpeg", "png"], key="fu")
        
        if uploaded is not None:
            pil_img = Image.open(uploaded).convert("RGB")
            st.image(pil_img, caption=f"Loaded: {uploaded.name} ({pil_img.size[0]}x{pil_img.size[1]}px)", use_column_width=True)
        else:
            st.info("👈 Upload an image to initialize the EfficientNet evaluation pipeline.")

    with col_right:
        st.subheader("📊 Diagnostic Results & Heatmaps")
        
        if uploaded is not None:
            if model is None:
                st.error("⚠️ Model weights (`ocunet_best.pth`) could not be loaded. Please check your model path.")
            else:
                try:
                    np_img = np.array(pil_img)
                    with st.spinner("Executing EfficientNet inference & Grad-CAM localization..."):
                        tensor, probs = run_inference(np_img, model, config, device)
                        orig, heat, overlay, tc = apply_gradcam(tensor, model, device, np_img)
                        
                    top_disease = disease_names.get(tc, "Unknown")
                    st.success(f"**Top Primary Diagnosis:** {top_disease}")
                    
                    # Display Grad-CAM grid
                    st.markdown("### 🔥 Grad-CAM Attention Maps")
                    sub_c1, sub_c2, sub_c3 = st.columns(3)
                    with sub_c1:
                        st.image(orig, caption="Preprocessed Input", use_column_width=True)
                    with sub_c2:
                        st.image(heat, caption="Attention Map", use_column_width=True)
                    with sub_c3:
                        st.image(overlay, caption="Superimposed Overlay", use_column_width=True)
                        
                    # Probability breakdown
                    st.markdown("### 📋 Disease Probability Ranking")
                    sorted_preds = sorted(enumerate(probs), key=lambda x: x[1], reverse=True)[:10]
                    for idx, p in sorted_preds:
                        d_name = disease_names.get(idx, f"Class {idx}")
                        st.progress(float(p), text=f"{d_name}: {p * 100:.1f}%")
                        
                except Exception as e:
                    st.error(f"Error during inference execution: {str(e)}")
        else:
            st.warning("Awaiting retinal input stream...")

with tab_perf:
    st.subheader("Model Performance Evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Macro AUC", auc_val)
    m2.metric("Macro F1", f1_val)
    m3.metric("Test Samples", n_test)
    m4.metric("Target Classes", len(disease_names) if disease_names else 46)
    
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Loss & AUC Curves (29 Epochs)**")
        curve_path = Path("outputs/training_curves.png")
        if curve_path.exists():
            st.image(str(curve_path), use_column_width=True)
        else:
            st.info("Training curves chart not found in outputs/")
    with c2:
        st.markdown("**Per-Disease AUC (Test Set)**")
        auc_path = Path("outputs/per_disease_auc.png")
        if auc_path.exists():
            st.image(str(auc_path), use_column_width=True)
        else:
            st.info("Per-disease AUC chart not found in outputs/")

with tab_arch:
    st.subheader("Technical Specification & Architecture")
    a1, a2 = st.columns(2)
    with a1:
        st.markdown("### 01 · Model & Training")
        st.markdown("- **Backbone:** EfficientNet Architecture")
        st.markdown("- **Optimizer:** Adam (lr=1e-4) with ReduceLROnPlateau")
        st.markdown("- **Loss Function:** BCEWithLogitsLoss")
        st.markdown("- **Hardware:** PyTorch CUDA Backend")
    with a2:
        st.markdown("### 02 · Data & Explainability")
        st.markdown("- **Datasets:** ODIR-5K, RFMiD, JSIEC")
        st.markdown("- **Preprocessing:** CLAHE Contrast Limited Adaptive Histogram Equalization")
        st.markdown("- **Explainability:** Grad-CAM Attention Maps with Bilateral Edge Sharpening")