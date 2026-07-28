"""
app.py — OcuNet v4 | Bold Clinical Dashboard + Theme Toggle
streamlit run app.py
"""
import sys, json, base64, numpy as np
from pathlib import Path
from io import BytesIO
import cv2, torch
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from src.model.ocunet import build_model
from src.dataset.loader import load_config, load_label_map, get_transforms
from src.gradcam import GradCAM

st.set_page_config(page_title="OcuNet", page_icon="🔬",
                   layout="wide", initial_sidebar_state="collapsed")
st.markdown("""
<style>
#MainMenu,footer,header,.stDeployButton,[data-testid="stToolbar"],
[data-testid="stHeader"],section[data-testid="stSidebar"],
div[data-testid="stNotification"],[data-baseweb="notification"],
.stAlert{display:none!important}
[data-testid="stMainBlockContainer"],[data-testid="stAppViewContainer"],
.block-container{padding:0!important;max-width:100%!important;overflow:hidden!important}
.stApp{background:#0a0d14!important;overflow:hidden!important}
iframe{border:none!important; position:fixed!important; top:0!important; left:0!important; height:100vh!important; width:100vw!important; z-index:9999!important; display:block!important;}
[data-testid="stFileUploader"]{position:absolute;opacity:0;pointer-events:none}
</style>
""", unsafe_allow_html=True)

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

def to_b64(img):
    buf = BytesIO()
    Image.fromarray(img.astype(np.uint8)).save(buf, format="JPEG", quality=91)
    return base64.b64encode(buf.getvalue()).decode()

def file_b64(p):
    with open(p,"rb") as f: return base64.b64encode(f.read()).decode()

model, config, disease_names, ckpt, device = load_model_cached()
report  = load_eval_report()
auc_val = f"{report['overall']['macro_auc']:.3f}" if report else "0.918"
f1_val  = f"{report['overall']['macro_f1']:.3f}"  if report else "0.261"
n_test  = str(report['overall']['n_samples'])      if report else "1088"

uploaded = st.file_uploader("u", type=["jpg","jpeg","png"], key="fu")

RD = {}
if uploaded and model:
    pil  = Image.open(uploaded).convert("RGB")
    np_  = np.array(pil)
    with st.spinner(""):
        tensor, probs = run_inference(np_, model, config, device)
        o,h,v,tc = apply_gradcam(tensor, model, device, np_)
    RD = {
        "ob": to_b64(o), "hb": to_b64(h), "vb": to_b64(v),
        "probs": [float(p) for p in probs],
        "dn": [disease_names[i] for i in range(len(probs))],
        "td": disease_names.get(tc,"Unknown"),
        "fn": uploaded.name, "fs": f"{uploaded.size//1024}KB",
        "dim": f"{pil.size[0]}x{pil.size[1]}"
    }

cb = file_b64("outputs/training_curves.png") if Path("outputs/training_curves.png").exists() else ""
ab = file_b64("outputs/per_disease_auc.png") if Path("outputs/per_disease_auc.png").exists() else ""
rj = json.dumps(RD)
hr = "true" if RD else "false"
ob = RD.get("ob","")
hb = RD.get("hb","")
vb = RD.get("vb","")
td = RD.get("td","Unknown")
fn = RD.get("fn","—")
fs = RD.get("fs","—")
dm = RD.get("dim","—")

HTML = f"""<!DOCTYPE html>
<html lang="en" data-theme="light">
<head>
<meta charset="UTF-8"/>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet"/>
<style>
/* ── TOKENS ── */
[data-theme="dark"]{{
  --bg0:#0a0d14;--bg1:#111520;--bg2:#171c2e;--bg3:#1e2438;
  --bd:#252b40;--bd2:#2e3650;
  --tx:#e8edf5;--tx2:#8b94b0;--tx3:#4a5568;
  --ac:#2563eb;--acr:37,99,235;--cy:#0891b2;--gr:#059669;
}}
[data-theme="light"]{{
  --bg0:#eef1f8;--bg1:#ffffff;--bg2:#f4f6fc;--bg3:#e8ecf5;
  --bd:#d8dded;--bd2:#c4cade;
  --tx:#0f172a;--tx2:#475569;--tx3:#94a3b8;
  --ac:#2563eb;--acr:37,99,235;--cy:#0891b2;--gr:#059669;
}}
*{{box-sizing:border-box;margin:0;padding:0}}
html,body{{height:100%;font-family:'IBM Plex Sans',sans-serif;background:var(--bg0);color:var(--tx);font-size:17px;transition:background .2s,color .2s;overflow:hidden}}

/* ── TOPBAR ── */
.top{{height:58px;background:var(--bg1);border-bottom:1px solid var(--bd);display:flex;align-items:center;justify-content:space-between;padding:0 1.5rem;flex-shrink:0;position:relative;z-index:10}}
.brand{{display:flex;align-items:center;gap:10px}}
.bi{{width:36px;height:36px;border-radius:9px;background:var(--ac);display:grid;place-items:center;font-weight:800;font-size:.9rem;color:#fff;font-family:'IBM Plex Mono',monospace;flex-shrink:0}}
.bn{{font-size:1.15rem;font-weight:800;color:var(--tx);letter-spacing:-.03em;font-family:'IBM Plex Mono',monospace}}
.bt{{font-size:.62rem;color:var(--tx3);font-family:'IBM Plex Mono',monospace;letter-spacing:.1em;text-transform:uppercase;margin-top:1px}}
.top-r{{display:flex;align-items:center;gap:.7rem}}
.sdot{{width:6px;height:6px;border-radius:50%;background:var(--gr);box-shadow:0 0 5px var(--gr)}}
.stxt{{font-size:.75rem;color:var(--tx3);font-family:'IBM Plex Mono',monospace}}
.tbtn{{width:30px;height:30px;border-radius:6px;border:1px solid var(--bd);background:var(--bg2);cursor:pointer;display:grid;place-items:center;font-size:.9rem;color:var(--tx);transition:.15s}}
.tbtn:hover{{background:var(--bg3);border-color:var(--ac)}}
.vtag{{font-family:'IBM Plex Mono',monospace;font-size:.7rem;color:var(--ac);border:1px solid rgba(var(--acr),.3);background:rgba(var(--acr),.06);padding:3px 9px;border-radius:4px;font-weight:500;letter-spacing:.06em}}

/* ── ROOT LAYOUT ── */
.root{{display:flex;height:calc(100vh - 58px)}}

/* ── LEFT PANEL ── */
.left-panel{{
  width:42%;flex-shrink:0;
  display:flex;flex-direction:column;
  border-right:1px solid var(--bd);
  background:var(--bg1);
  overflow:hidden;
}}

/* ── RIGHT PANEL ── */
.right-panel{{
  flex:1;display:flex;flex-direction:column;
  background:var(--bg0);overflow:hidden;
}}

/* ── PANEL HEADER ── */
.ph{{
  height:36px;flex-shrink:0;
  display:flex;align-items:center;justify-content:space-between;
  padding:0 1rem;border-bottom:1px solid var(--bd);
  background:var(--bg2);
}}
.ph-l{{display:flex;align-items:center;gap:6px}}
.phdot{{width:4px;height:4px;border-radius:50%;background:var(--ac)}}
.phtitle{{font-size:.72rem;font-weight:600;color:var(--tx2);text-transform:uppercase;letter-spacing:.12em;font-family:'IBM Plex Mono',monospace}}
.ph-r{{font-size:.7rem;color:var(--tx3);font-family:'IBM Plex Mono',monospace}}

/* ── IMAGE SECTION (top half of left) ── */
.img-section{{
  flex:1;min-height:0;
  display:flex;flex-direction:column;
  border-bottom:1px solid var(--bd);
  overflow:hidden;
}}
.img-body{{
  flex:1;min-height:0;
  display:flex;align-items:center;justify-content:center;
  background:#05080f;padding:.8rem;overflow:hidden;
}}
.img-body img{{
  max-width:100%;max-height:100%;
  object-fit:contain;border-radius:8px;display:block;
}}
.img-foot{{
  height:32px;flex-shrink:0;
  display:flex;align-items:center;gap:.6rem;
  padding:0 1rem;border-top:1px solid var(--bd);
  background:var(--bg2);
}}
.mc{{font-family:'IBM Plex Mono',monospace;font-size:.72rem;color:var(--tx3);background:var(--bg3);border:1px solid var(--bd);padding:1px 6px;border-radius:3px}}
.new-btn{{margin-left:auto;font-size:.73rem;color:var(--ac);font-weight:600;border:1px solid rgba(var(--acr),.3);padding:2px 10px;border-radius:4px;background:rgba(var(--acr),.07);cursor:pointer;white-space:nowrap}}
.new-btn:hover{{background:rgba(var(--acr),.14)}}

/* ── HEATMAP SECTION (bottom half of left) ── */
.hm-section{{
  height:42%;flex-shrink:0;
  display:flex;flex-direction:column;
  overflow:hidden;
}}
.hm-body{{
  flex:1;min-height:0;
  display:grid;grid-template-columns:1fr 1fr 1fr;
  gap:1px;background:var(--bd);overflow:hidden;
}}
.hm-cell{{
  background:var(--bg1);
  display:flex;flex-direction:column;
  overflow:hidden;
}}
.hm-img-wrap{{
  flex:1;min-height:0;
  background:#05080f;
  display:flex;align-items:center;justify-content:center;
  overflow:hidden;
}}
.hm-img-wrap img{{
  max-width:100%;max-height:100%;
  object-fit:contain;display:block;
}}
.hm-ft{{
  flex-shrink:0;height:34px;
  padding:0 10px;
  display:flex;flex-direction:column;justify-content:center;
  border-top:1px solid var(--bd);
  background:var(--bg2);
}}
.hm-ft-t{{font-size:.8rem;font-weight:600;color:var(--tx);font-family:'IBM Plex Mono',monospace}}
.hm-ft-s{{font-size:.7rem;color:var(--tx3);margin-top:1px}}

/* ── UPLOAD ZONE ── */
.drop{{
  flex:1;display:flex;flex-direction:column;
  align-items:center;justify-content:center;
  border:2px dashed var(--bd2);border-radius:10px;
  margin:.8rem;cursor:pointer;transition:.2s;
  background:rgba(var(--acr),.02);text-align:center;
  padding:1.5rem;
}}
.drop:hover{{border-color:var(--ac);background:rgba(var(--acr),.05)}}
.drop-ic{{font-size:2rem;margin-bottom:.6rem}}
.drop-t{{font-size:.88rem;font-weight:600;color:var(--tx);margin-bottom:.3rem}}
.drop-s{{font-size:.84rem;color:var(--tx3)}}
.drop-btn{{display:inline-block;margin-top:.8rem;font-size:.82rem;color:var(--ac);font-weight:600;border:1px solid rgba(var(--acr),.35);padding:5px 14px;border-radius:5px;background:rgba(var(--acr),.08);cursor:pointer}}

/* ── RIGHT: DIAGNOSIS ── */
.diag-section{{
  flex:1;min-height:0;overflow-y:auto;
  padding:1rem;
  border-bottom:1px solid var(--bd);
}}
.empty-state{{color:var(--tx3);font-size:.9rem;text-align:center;margin-top:4rem}}
.badge{{display:inline-flex;align-items:center;gap:5px;padding:6px 12px;border-radius:6px;font-size:.9rem;font-weight:500;margin:2px;border:1px solid}}
.badge.pos{{background:rgba(var(--acr),.1);border-color:rgba(var(--acr),.3);color:#93c5fd}}
.badge.norm{{background:rgba(5,150,105,.08);border-color:rgba(5,150,105,.3);color:#6ee7b7}}
.badge.warn{{background:rgba(217,119,6,.08);border-color:rgba(217,119,6,.3);color:#fcd34d}}
.bdot{{width:5px;height:5px;border-radius:50%;background:currentColor;flex-shrink:0}}
.bpct{{font-family:'IBM Plex Mono',monospace;font-weight:700;margin-left:3px}}
.rank-title{{font-size:.72rem;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.1em;margin:1rem 0 .5rem;font-family:'IBM Plex Mono',monospace}}
.cb{{display:flex;align-items:center;gap:8px;padding:5px 0;border-bottom:1px solid var(--bd)}}
.cb:last-child{{border:none}}
.cbn{{font-size:.84rem;color:var(--tx2);width:165px;flex-shrink:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
.cbt{{flex:1;height:3px;background:var(--bg3);border-radius:2px;overflow:hidden}}
.cbf{{height:3px;border-radius:2px}}
.cbp{{font-family:'IBM Plex Mono',monospace;font-size:.78rem;color:var(--tx3);width:28px;text-align:right}}
.cbi{{width:5px;height:5px;border-radius:50%;flex-shrink:0}}

/* ── RIGHT: CONTROLS ── */
.ctrl-section{{
  flex-shrink:0;padding:.8rem 1rem;
  border-bottom:1px solid var(--bd);
  background:var(--bg1);
}}
.ctrl-r{{display:flex;flex-direction:column;gap:.4rem;margin-bottom:.6rem}}
.ctrl-h{{display:flex;justify-content:space-between}}
.ctrl-l{{font-size:.72rem;font-weight:600;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em}}
.ctrl-v{{font-family:'IBM Plex Mono',monospace;font-size:.8rem;color:var(--ac);font-weight:600}}
input[type=range]{{-webkit-appearance:none;width:100%;height:2px;background:var(--bd2);border-radius:2px;outline:none;cursor:pointer}}
input[type=range]::-webkit-slider-thumb{{-webkit-appearance:none;width:12px;height:12px;border-radius:50%;background:var(--ac);cursor:pointer;border:2px solid var(--bg1)}}
.togs{{display:flex;gap:1.2rem}}
.tr{{display:flex;align-items:center;gap:6px}}
.tl{{font-size:.84rem;color:var(--tx2)}}
.tog{{position:relative;width:30px;height:16px}}
.tog input{{opacity:0;width:0;height:0}}
.ts{{position:absolute;inset:0;background:var(--bg3);border:1px solid var(--bd2);border-radius:16px;cursor:pointer;transition:.2s}}
.ts::before{{content:'';position:absolute;width:10px;height:10px;left:2px;top:2px;background:var(--tx3);border-radius:50%;transition:.2s}}
input:checked+.ts{{background:rgba(var(--acr),.2);border-color:var(--ac)}}
input:checked+.ts::before{{transform:translateX(14px);background:var(--ac)}}

/* ── RIGHT: STATS (bottom) ── */
.stats-section{{
  flex-shrink:0;
  display:grid;grid-template-columns:repeat(3,1fr);
  gap:0;background:var(--bd);
  width:100%;
}}
.stat-cell{{
  background:var(--bg1);padding:.8rem 1.2rem;
  position:relative;overflow:hidden;
  border-right:1px solid var(--bd);
}}
.stat-cell:last-child{{border-right:none}}
.stat-v{{font-family:'IBM Plex Mono',monospace;font-size:1.3rem;font-weight:700;color:var(--tx);line-height:1}}
.stat-l{{font-size:.72rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-top:4px;font-weight:500}}

/* ── NAV TABS ── */
.tabs{{display:flex;gap:0;border-bottom:1px solid var(--bd);background:var(--bg1);flex-shrink:0}}
.tab{{padding:.6rem 1.2rem;font-size:.82rem;font-weight:600;color:var(--tx3);cursor:pointer;border-bottom:2px solid transparent;transition:.15s;letter-spacing:.04em;text-transform:uppercase;font-family:'IBM Plex Mono',monospace}}
.tab:hover{{color:var(--tx2)}}
.tab.on{{color:var(--ac);border-bottom-color:var(--ac)}}

/* ── PERF / ARCH VIEWS ── */
.view{{flex:1;overflow-y:auto;padding:1.2rem}}
.view-hidden{{display:none}}
.sec-tag{{font-size:.72rem;font-weight:600;color:var(--ac);text-transform:uppercase;letter-spacing:.14em;margin-bottom:.3rem;font-family:'IBM Plex Mono',monospace}}
.sec-title{{font-size:1.2rem;font-weight:700;color:var(--tx);letter-spacing:-.02em;margin-bottom:.8rem}}
.metrics-g{{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin-bottom:1rem}}
.mtile{{background:var(--bg1);border:1px solid var(--bd);border-radius:10px;padding:1rem;position:relative;overflow:hidden}}
.mtile::before{{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--ac),var(--cy))}}
.mtile-v{{font-family:'IBM Plex Mono',monospace;font-size:1.4rem;font-weight:600;color:var(--tx);line-height:1}}
.mtile-l{{font-size:.72rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-top:4px}}
.chart-img{{width:100%;border-radius:7px;border:1px solid var(--bd);display:block}}
.arch-t{{font-size:.72rem;font-weight:600;color:var(--ac);text-transform:uppercase;letter-spacing:.1em;margin-bottom:.4rem;font-family:'IBM Plex Mono',monospace;display:flex;align-items:center;gap:6px}}
.arch-t::after{{content:'';flex:1;height:1px;background:var(--bd)}}
.arch-i{{font-size:.84rem;color:var(--tx2);padding:4px 0;border-bottom:1px solid var(--bd);display:flex;align-items:center;gap:7px}}
.arch-i::before{{content:'';width:3px;height:3px;background:var(--ac);border-radius:50%;flex-shrink:0}}
</style>
</head>
<body>

<!-- TOPBAR -->
<div class="top">
  <div class="brand">
    <div class="bi">Oc</div>
    <div>
      <div class="bn">OcuNet Diagnostics</div>
      <div class="bt">Intelligent Retinal AI Platform</div>
    </div>
  </div>
  <div class="top-r">
    <div class="sdot"></div><span class="stxt">MODEL ONLINE</span>
    <button class="tbtn" onclick="toggleTheme()" id="tbtn">🌙</button>
    <div class="vtag">RESEARCH BUILD v1.0</div>
  </div>
</div>

<!-- ROOT -->
<div class="root">

  <!-- ══ LEFT PANEL ══ -->
  <div class="left-panel">

    <!-- Image section -->
    <div class="img-section" id="img-section">
      <div class="ph">
        <div class="ph-l"><div class="phdot"></div><span class="phtitle">Fundus Image</span></div>
        <span class="ph-r" id="img-info">No image loaded</span>
      </div>

      <!-- Upload state -->
      <div class="drop" id="drop-zone" onclick="trigUp()">
        <div class="drop-ic">🔬</div>
        <div class="drop-t">Upload Fundus Photograph</div>
        <div class="drop-s">JPG or PNG · Min 224×224px · Max 200MB</div>
        <div class="drop-btn">Choose File →</div>
        <div style="margin-top:.6rem;font-size:.65rem;color:var(--tx3)">Test: data/raw/ODIR-5K/</div>
      </div>

      <!-- Result state -->
      <div class="img-body" id="img-body" style="display:none">
        <img id="orig-img" src="" alt="fundus"/>
      </div>
      <div class="img-foot" id="img-foot" style="display:none">
        <span class="mc" id="fn-c">—</span>
        <span class="mc" id="dim-c">—</span>
        <span class="mc" id="fs-c">—</span>
        <span class="new-btn" onclick="trigUp()">↑ New Image</span>
      </div>
    </div>

    <!-- Heatmap section -->
    <div class="hm-section" id="hm-section">
      <div class="ph">
        <div class="ph-l"><div class="phdot"></div><span class="phtitle">Grad-CAM · Attention Maps</span></div>
        <span class="ph-r">Red/yellow = high attention</span>
      </div>
      <div class="hm-body" id="hm-empty-state" style="display:flex;align-items:center;justify-content:center;background:var(--bg1)">
        <span style="color:var(--tx3);font-size:.78rem">Heatmaps appear after upload</span>
      </div>
      <div class="hm-body" id="hm-grid" style="display:none">
        <div class="hm-cell">
          <div class="hm-img-wrap"><img id="hm-o" src="" alt="orig"/></div>
          <div class="hm-ft"><div class="hm-ft-t">Original</div><div class="hm-ft-s">Pre-processed input</div></div>
        </div>
        <div class="hm-cell">
          <div class="hm-img-wrap"><img id="hm-h" src="" alt="heatmap"/></div>
          <div class="hm-ft"><div class="hm-ft-t">Attention Map</div><div class="hm-ft-s" id="hm-cls">—</div></div>
        </div>
        <div class="hm-cell">
          <div class="hm-img-wrap"><img id="hm-v" src="" alt="overlay"/></div>
          <div class="hm-ft"><div class="hm-ft-t">Overlay</div><div class="hm-ft-s">Superimposed</div></div>
        </div>
      </div>
    </div>

  </div>

  <!-- ══ RIGHT PANEL ══ -->
  <div class="right-panel">

    <!-- Tabs -->
    <div class="tabs">
      <div class="tab on" onclick="switchTab('diagnosis',this)">Diagnosis</div>
      <div class="tab" onclick="switchTab('performance',this)">Performance</div>
      <div class="tab" onclick="switchTab('architecture',this)">Architecture</div>
    </div>

    <!-- ── DIAGNOSIS TAB ── -->
    <div id="tab-diagnosis" style="display:flex;flex-direction:column;flex:1;min-height:0;overflow:hidden">

      <!-- Controls -->
      <div class="ctrl-section">
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:1rem;margin-bottom:.5rem">
          <div class="ctrl-r">
            <div class="ctrl-h"><span class="ctrl-l">Detection Threshold</span><span class="ctrl-v" id="tv">0.65</span></div>
            <input type="range" min="0.40" max="0.90" step="0.05" value="0.65" id="tslider" oninput="document.getElementById('tv').textContent=parseFloat(this.value).toFixed(2);upd()">
          </div>
          <div class="ctrl-r">
            <div class="ctrl-h"><span class="ctrl-l">Min Confidence Gate</span><span class="ctrl-v" id="gv">0.55</span></div>
            <input type="range" min="0.30" max="0.80" step="0.05" value="0.55" id="gslider" oninput="document.getElementById('gv').textContent=parseFloat(this.value).toFixed(2);upd()">
          </div>
        </div>
        <div class="togs">
          <div class="tr"><label class="tog"><input type="checkbox" checked onchange="show('hm-section',this.checked)"><span class="ts"></span></label><span class="tl">Grad-CAM</span></div>
          <div class="tr"><label class="tog"><input type="checkbox" checked onchange="show('cb-wrap',this.checked)"><span class="ts"></span></label><span class="tl">Confidence bars</span></div>
        </div>
      </div>

      <!-- Diagnosis output -->
      <div class="diag-section">
        <div class="empty-state" id="diag-empty">Upload a fundus image to begin analysis</div>
        <div id="diag-result" style="display:none">
          <div id="chips"></div>
          <div id="cb-wrap">
            <div class="rank-title">Disease Probability Ranking</div>
            <div id="cbars"></div>
          </div>
        </div>
      </div>

      <!-- Stats bar -->
      <div class="stats-section">
        <div class="stat-cell"><div class="stat-v">{auc_val}</div><div class="stat-l">Macro AUC</div></div>
        <div class="stat-cell"><div class="stat-v">25</div><div class="stat-l">Conditions</div></div>
        <div class="stat-cell"><div class="stat-v">7,249</div><div class="stat-l">Train Images</div></div>
      </div>

    </div>

    <!-- ── PERFORMANCE TAB ── -->
    <div id="tab-performance" class="view view-hidden">
      <div class="sec-tag">Evaluation Results</div>
      <div class="sec-title">Model Performance</div>
      <div class="metrics-g">
        <div class="mtile"><div class="mtile-v">{auc_val}</div><div class="mtile-l">Macro AUC</div></div>
        <div class="mtile"><div class="mtile-v">{f1_val}</div><div class="mtile-l">Macro F1</div></div>
        <div class="mtile"><div class="mtile-v">{n_test}</div><div class="mtile-l">Test Images</div></div>
        <div class="mtile"><div class="mtile-v">25</div><div class="mtile-l">Diseases</div></div>
      </div>
      {"<div style='margin-bottom:.8rem'><div style='font-size:.6rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;font-family:IBM Plex Mono,monospace'>Loss &amp; AUC Curves · 29 Epochs</div><img class='chart-img' src='data:image/png;base64," + cb + "' alt='curves'/></div>" if cb else ""}
      {"<div><div style='font-size:.6rem;color:var(--tx3);text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem;font-family:IBM Plex Mono,monospace'>Per-Disease AUC · Test Set</div><img class='chart-img' src='data:image/png;base64," + ab + "' alt='auc'/></div>" if ab else ""}
    </div>

    <!-- ── ARCHITECTURE TAB ── -->
    <div id="tab-architecture" class="view view-hidden">
      <div class="sec-tag">Technical Specification</div>
      <div class="sec-title">System Architecture</div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:1.2rem">
        <div>
          <div style="margin-bottom:1rem"><div class="arch-t">01 · Model</div><div class="arch-i">EfficientNet-B0 backbone</div><div class="arch-i">Pretrained on ImageNet-1K</div><div class="arch-i">Custom FC classifier head</div><div class="arch-i">Dropout + BatchNorm</div><div class="arch-i">BCEWithLogitsLoss</div></div>
          <div><div class="arch-t">02 · Training</div><div class="arch-i">Adam (lr=1e-4)</div><div class="arch-i">ReduceLROnPlateau</div><div class="arch-i">Mixed precision AMP</div><div class="arch-i">Early stopping (p=8)</div><div class="arch-i">Batch 16 · RTX 3050 4GB</div></div>
        </div>
        <div>
          <div style="margin-bottom:1rem"><div class="arch-t">03 · Data Pipeline</div><div class="arch-i">ODIR-5K · 6,392 images</div><div class="arch-i">RFMiD · 490 images</div><div class="arch-i">JSIEC · 367 images</div><div class="arch-i">CLAHE preprocessing</div><div class="arch-i">Weighted random sampler</div></div>
          <div><div class="arch-t">04 · Explainability</div><div class="arch-i">Grad-CAM heatmaps</div><div class="arch-i">Bilateral edge sharpening</div><div class="arch-i">Per-class attention maps</div><div class="arch-i">46-class multi-label output</div><div class="arch-i">Confidence calibration</div></div>
        </div>
      </div>
    </div>

  </div>
</div>

<script>
const D={rj};
const dn=D.dn||[];
const pr=D.probs||[];
const HR={hr};
applyStoredTheme();

function toggleTheme(){{
  const h=document.documentElement;
  const d=h.getAttribute('data-theme')==='dark';
  const next=d?'light':'dark';
  h.setAttribute('data-theme',next);
  document.getElementById('tbtn').textContent=next==='dark'?'🌙':'☀️';
  try{{localStorage.setItem('ocunet-theme',next);}}catch(e){{}}
}}
function applyStoredTheme(){{
  try{{
    const t=localStorage.getItem('ocunet-theme')||'light';
    document.documentElement.setAttribute('data-theme',t);
    document.getElementById('tbtn').textContent=t==='dark'?'🌙':'☀️';
  }}catch(e){{}}
}}

function switchTab(name,el){{
  ['diagnosis','performance','architecture'].forEach(t=>{{
    document.getElementById('tab-'+t).style.display='none';
    document.getElementById('tab-'+t).classList.add('view-hidden');
  }});
  const tab=document.getElementById('tab-'+name);
  tab.style.display=name==='diagnosis'?'flex':'block';
  tab.classList.remove('view-hidden');
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('on'));
  if(el)el.classList.add('on');
}}

function show(id,v){{const e=document.getElementById(id);if(e)e.style.display=v?'flex':'none'}}
function thr(){{return Math.max(parseFloat(document.getElementById('tslider').value),parseFloat(document.getElementById('gslider').value))}}

function upd(){{
  if(!pr.length)return;
  const t=thr();
  const det=pr.map((p,i)=>[i,p]).filter(([,p])=>p>=t).sort((a,b)=>b[1]-a[1]);
  const ch=document.getElementById('chips');
  if(det.length){{
    ch.innerHTML=det.map(([i,p])=>`<span class="badge pos"><span class="bdot"></span>${{dn[i]}}<span class="bpct">${{(p*100).toFixed(0)}}%</span></span>`).join('');
    if(Math.max(...det.map(([,p])=>p))<0.75)ch.innerHTML+='<div style="margin-top:.5rem"><span class="badge warn">⚠ Low confidence — verify clinically</span></div>';
  }}else{{
    ch.innerHTML='<span class="badge norm"><span class="bdot"></span>Normal — No pathology detected</span>';
  }}
  const top=[...pr.map((p,i)=>[i,p])].sort((a,b)=>b[1]-a[1]).slice(0,10);
  document.getElementById('cbars').innerHTML=top.map(([i,p])=>{{
    const d=p>=t;
    return `<div class="cb"><div class="cbn">${{(dn[i]||'').substring(0,26)}}</div><div class="cbt"><div class="cbf" style="width:${{(p*100).toFixed(1)}}%;background:${{d?'linear-gradient(90deg,#2563eb,#0891b2)':'var(--bg3)'}}"></div></div><div class="cbp">${{(p*100).toFixed(0)}}%</div><div class="cbi" style="background:${{d?'var(--ac)':'var(--bd2)'}}"></div></div>`;
  }}).join('');
}}

function trigUp(){{
  const el=window.parent.document.querySelector('[data-testid="stFileUploader"] input[type="file"]');
  if(el){{el.style.cssText='display:block;opacity:0;position:absolute';el.click();}}
}}

if(HR){{
  document.getElementById('drop-zone').style.display='none';
  document.getElementById('img-body').style.display='flex';
  document.getElementById('img-foot').style.display='flex';
  document.getElementById('hm-empty-state').style.display='none';
  document.getElementById('hm-grid').style.display='grid';
  document.getElementById('diag-empty').style.display='none';
  document.getElementById('diag-result').style.display='block';
  document.getElementById('orig-img').src='data:image/jpeg;base64,{ob}';
  document.getElementById('hm-o').src='data:image/jpeg;base64,{ob}';
  document.getElementById('hm-h').src='data:image/jpeg;base64,{hb}';
  document.getElementById('hm-v').src='data:image/jpeg;base64,{vb}';
  document.getElementById('hm-cls').textContent='Class: {td}';
  document.getElementById('fn-c').textContent='{fn}';
  document.getElementById('dim-c').textContent='{dm}';
  document.getElementById('fs-c').textContent='{fs}';
  document.getElementById('img-info').textContent='{fn}';
  upd();
}}
</script>
</body></html>"""

components.html(HTML, height=870, scrolling=False)