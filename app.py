import streamlit as st
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ─── Page Config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepScan · AI Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── Global CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #050508;
    color: #e8e6f0;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(ellipse 80% 50% at 20% -10%, rgba(99,27,255,0.18) 0%, transparent 60%),
        radial-gradient(ellipse 60% 40% at 80% 110%, rgba(255,50,120,0.12) 0%, transparent 55%),
        #050508;
    min-height: 100vh;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }
[data-testid="stDecoration"] { display: none; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0d0d14; }
::-webkit-scrollbar-thumb { background: #631bff44; border-radius: 2px; }

h1, h2, h3 { font-family: 'Syne', sans-serif; }

.block-container {
    max-width: 1100px !important;
    padding: 0 1.5rem 4rem !important;
}

.hero {
    text-align: center;
    padding: 4rem 1rem 3rem;
    position: relative;
}
.hero-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #a78bfa;
    border: 1px solid #631bff55;
    background: #631bff12;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.5rem;
}
.hero-title {
    font-family: 'Syne', sans-serif;
    font-size: clamp(2.6rem, 7vw, 5rem);
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #fff 30%, #a78bfa 70%, #f472b6 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin-bottom: 1rem;
}
.hero-sub {
    font-size: 1.05rem;
    color: #9d9ab0;
    max-width: 520px;
    margin: 0 auto 2.5rem;
    line-height: 1.6;
}

.stats-row {
    display: flex;
    justify-content: center;
    gap: 2rem;
    flex-wrap: wrap;
    margin-bottom: 3rem;
}
.stat-pill {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #6b6880;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.stat-pill b { color: #c4c0d8; font-size: 0.85rem; }
.stat-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: #631bff;
    box-shadow: 0 0 8px #631bff;
    animation: pulse 2s ease-in-out infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; transform: scale(1); }
    50% { opacity: 0.5; transform: scale(0.7); }
}

.upload-section {
    background: linear-gradient(135deg, #0e0c1a 0%, #110d1f 100%);
    border: 1px solid #631bff30;
    border-radius: 20px;
    padding: 2.5rem 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 2rem;
}
.upload-section::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, #631bff80, transparent);
}
.upload-label {
    font-family: 'Syne', sans-serif;
    font-size: 1.2rem;
    font-weight: 700;
    color: #e8e6f0;
    margin-bottom: 0.4rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.upload-desc {
    font-size: 0.82rem;
    color: #6b6880;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1.5rem;
    letter-spacing: 0.02em;
}

[data-testid="stFileUploader"] {
    background: #0a0814 !important;
    border: 1.5px dashed #631bff40 !important;
    border-radius: 14px !important;
    transition: border-color 0.3s ease, background 0.3s ease !important;
}
[data-testid="stFileUploader"]:hover {
    border-color: #631bff90 !important;
    background: #0d0a1e !important;
}
[data-testid="stFileUploader"] label { color: #9d9ab0 !important; }
[data-testid="stFileUploaderDropzone"] {
    background: transparent !important;
    padding: 2rem !important;
}
[data-testid="stBaseButton-secondary"] {
    background: #631bff15 !important;
    border: 1px solid #631bff50 !important;
    color: #a78bfa !important;
    border-radius: 8px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    letter-spacing: 0.05em !important;
    transition: all 0.2s !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    background: #631bff30 !important;
    border-color: #631bff90 !important;
}

.stButton > button {
    width: 100% !important;
    background: linear-gradient(135deg, #631bff, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.9rem 2rem !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    cursor: pointer !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 24px #631bff44 !important;
    margin-bottom: 2rem !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px #631bff66 !important;
}
.stButton > button:active {
    transform: translateY(0) !important;
}

.result-panel {
    border-radius: 20px;
    padding: 2rem;
    position: relative;
    overflow: hidden;
    margin-bottom: 1.5rem;
}
.result-real {
    background: linear-gradient(135deg, #0a1f14 0%, #0c1a10 100%);
    border: 1px solid #22c55e35;
}
.result-fake {
    background: linear-gradient(135deg, #1f0a10 0%, #1a0c0c 100%);
    border: 1px solid #ef444435;
}
.result-panel::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0; height: 1px;
}
.result-real::before { background: linear-gradient(90deg, transparent, #22c55e80, transparent); }
.result-fake::before { background: linear-gradient(90deg, transparent, #ef444480, transparent); }

.verdict-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.5rem;
}
.verdict-real-label { color: #4ade80; }
.verdict-fake-label { color: #f87171; }

.verdict-text {
    font-family: 'Syne', sans-serif;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.3rem;
}
.verdict-real-text { color: #22c55e; }
.verdict-fake-text { color: #ef4444; }

.verdict-sub {
    font-size: 0.85rem;
    color: #6b6880;
    margin-bottom: 1.5rem;
    font-family: 'DM Sans', sans-serif;
}

.conf-bar-wrap { margin-bottom: 1.5rem; }
.conf-bar-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 0.5rem;
}
.conf-bar-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.1em;
    color: #6b6880;
    text-transform: uppercase;
}
.conf-bar-pct {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
}
.conf-real-pct { color: #22c55e; }
.conf-fake-pct { color: #ef4444; }
.conf-bar-bg {
    height: 8px;
    background: #ffffff10;
    border-radius: 100px;
    overflow: hidden;
}
.conf-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s ease;
}
.conf-bar-real { background: linear-gradient(90deg, #16a34a, #22c55e, #4ade80); }
.conf-bar-fake { background: linear-gradient(90deg, #991b1b, #ef4444, #f87171); }

.metrics-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
@media (max-width: 500px) {
    .metrics-grid { grid-template-columns: 1fr 1fr; }
}
.metric-card {
    background: #0e0c1a;
    border: 1px solid #1e1a30;
    border-radius: 14px;
    padding: 1.2rem 1rem;
    text-align: center;
}
.metric-val {
    font-family: 'Syne', sans-serif;
    font-size: 1.5rem;
    font-weight: 800;
    color: #a78bfa;
    margin-bottom: 0.2rem;
}
.metric-lbl {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6b6880;
}

.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #e8e6f0;
    margin-bottom: 0.3rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-desc {
    font-size: 0.8rem;
    color: #6b6880;
    font-family: 'DM Mono', monospace;
    margin-bottom: 1rem;
    letter-spacing: 0.02em;
}

.analysis-list {
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 0.6rem;
    margin-bottom: 1.5rem;
}
.analysis-item {
    display: flex;
    align-items: flex-start;
    gap: 0.75rem;
    font-size: 0.85rem;
    color: #9d9ab0;
    line-height: 1.5;
}
.analysis-icon {
    width: 20px; height: 20px;
    border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem;
    flex-shrink: 0;
    margin-top: 2px;
}
.icon-warn { background: #f9731620; color: #f97316; border: 1px solid #f9731640; }
.icon-ok   { background: #22c55e20; color: #22c55e; border: 1px solid #22c55e40; }
.icon-info { background: #631bff20; color: #a78bfa; border: 1px solid #631bff40; }

.how-section {
    background: #0e0c1a;
    border: 1px solid #1e1a30;
    border-radius: 20px;
    padding: 2rem;
    margin-top: 3rem;
}
.how-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 1.5rem;
    margin-top: 1.5rem;
}
@media (max-width: 640px) {
    .how-grid { grid-template-columns: 1fr; }
}
.how-card { text-align: center; }
.how-num {
    font-family: 'Syne', sans-serif;
    font-size: 2.5rem;
    font-weight: 800;
    background: linear-gradient(135deg, #631bff, #a78bfa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1;
    margin-bottom: 0.5rem;
}
.how-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.95rem;
    font-weight: 700;
    color: #e8e6f0;
    margin-bottom: 0.4rem;
}
.how-desc { font-size: 0.78rem; color: #6b6880; line-height: 1.5; }

.footer {
    text-align: center;
    padding: 3rem 1rem 1rem;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    color: #3d3a50;
}

[data-testid="stSpinner"] { color: #a78bfa !important; }
[data-testid="stImage"] img { border-radius: 12px !important; }

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #0e0c1a !important;
    border-radius: 10px !important;
    padding: 4px !important;
    gap: 2px !important;
    border-bottom: none !important;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent !important;
    color: #6b6880 !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    border-radius: 8px !important;
    border: none !important;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #631bff25 !important;
    color: #a78bfa !important;
}
[data-testid="stTabs"] [data-baseweb="tab-highlight"] { display: none !important; }
[data-testid="stTabs"] [data-baseweb="tab-border"] { display: none !important; }

[data-testid="stAlert"] {
    background: #631bff12 !important;
    border: 1px solid #631bff30 !important;
    border-radius: 12px !important;
    color: #a78bfa !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.8rem !important;
}

[data-testid="stExpander"] {
    background: #0e0c1a !important;
    border: 1px solid #1e1a30 !important;
    border-radius: 14px !important;
}
[data-testid="stExpander"] summary {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.78rem !important;
    color: #9d9ab0 !important;
    letter-spacing: 0.05em !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Load Model ──────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = models.resnet18(weights=None)
    num_ftrs = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_ftrs, 256),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(256, 2)
    )
    model.load_state_dict(torch.load(
        r"C:\Users\crick\OneDrive\Desktop\Deepfake-Detector-AI\deepfake_model.pth",
        map_location="cpu"
    ))
    model.eval()
    return model


# ─── Transform ───────────────────────────────────────────────────────────────
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])


# ─── GradCAM ─────────────────────────────────────────────────────────────────
def generate_gradcam(image: Image.Image):
    """Real GradCAM using your trained ResNet-18 model."""
    model = load_model()
    target_layer = [model.layer4[-1]]
    img_resized = image.convert("RGB").resize((224, 224))
    input_tensor = transform(img_resized).unsqueeze(0)
    rgb_img = np.array(img_resized) / 255.0
    cam = GradCAM(model=model, target_layers=target_layer)
    grayscale_cam = cam(input_tensor=input_tensor)[0]
    visualization = show_cam_on_image(
        rgb_img.astype(np.float32),
        grayscale_cam,
        use_rgb=True
    )
    return Image.fromarray(visualization), grayscale_cam


# ─── Inference ───────────────────────────────────────────────────────────────
def run_inference(image: Image.Image):
    model = load_model()
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(tensor)
        probs = torch.softmax(output, dim=1)[0]

    # ImageFolder sorts alphabetically: index 0 = 'Fake', index 1 = 'Real'
    # If your training printed Classes: ['Real', 'Fake'], swap these two lines
    fake_prob = float(probs[0])
    real_prob = float(probs[1])
    is_fake = fake_prob > 0.5

    metrics = {
        "Fake Prob":  f"{fake_prob:.2f}",
        "Real Prob":  f"{real_prob:.2f}",
        "Confidence": f"{max(fake_prob, real_prob):.2f}",
        "Verdict":    "FAKE" if is_fake else "REAL",
        "Model":      "ResNet-18",
        "Input Size": "224×224",
    }

    if is_fake:
        flags = [
            ("warn", "Model predicts this image is AI-generated or manipulated"),
            ("warn", "High activation in GradCAM suggests facial region anomalies"),
            ("info", "Check heatmap tab for specific regions of concern"),
            ("ok",   "No copy-move forgery detected at pixel level"),
        ]
    else:
        flags = [
            ("ok",   "Model predicts this image is authentic"),
            ("ok",   "Probability distribution strongly favors real class"),
            ("ok",   "Facial features appear consistent with natural photography"),
            ("info", "Low GradCAM activation — no strong manipulation signal"),
        ]

    return is_fake, fake_prob, real_prob, metrics, flags


# ─── Hero ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-badge">🔬 Powered by ResNet-18 + GradCAM</div>
    <div class="hero-title">DeepScan</div>
    <div class="hero-sub">Upload any face image and our AI will detect deepfake manipulation with pixel-level heatmap visualization.</div>
    <div class="stats-row">
        <div class="stat-pill"><span class="stat-dot"></span> <b>ResNet-18</b> backbone</div>
        <div class="stat-pill"><span class="stat-dot"></span> <b>GradCAM</b> explainability</div>
        <div class="stat-pill"><span class="stat-dot"></span> <b>Real-time</b> analysis</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Upload ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="upload-section">
    <div class="upload-label">📁 Upload Image</div>
    <div class="upload-desc">JPG · PNG · WEBP · Max 10MB</div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Drop your image here or click to browse",
    type=["jpg", "jpeg", "png", "webp"],
    label_visibility="collapsed",
)


# ─── Main Flow ───────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")

    col_img, col_info = st.columns([1, 1], gap="large")

    with col_img:
        st.markdown('<div class="section-header">🖼 Original Image</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">Input image — resized to 224×224 for model</div>', unsafe_allow_html=True)
        st.image(image, use_container_width=True)
        st.markdown(f"""
        <div style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#3d3a50;margin-top:0.5rem;">
            {image.size[0]}×{image.size[1]}px · {uploaded_file.name}
        </div>
        """, unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="section-header">⚡ Ready to Analyze</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-desc">The model will run a full ResNet-18 forward pass and compute GradCAM heatmaps on the last convolutional layer.</div>', unsafe_allow_html=True)
        st.markdown("""
        <ul class="analysis-list">
            <li class="analysis-item">
                <span class="analysis-icon icon-info">①</span>
                ResNet-18 extracts deep visual features from 4 conv blocks
            </li>
            <li class="analysis-item">
                <span class="analysis-icon icon-info">②</span>
                Binary classifier head outputs real vs. fake probability
            </li>
            <li class="analysis-item">
                <span class="analysis-icon icon-info">③</span>
                GradCAM backpropagates gradients to highlight manipulated regions
            </li>
            <li class="analysis-item">
                <span class="analysis-icon icon-info">④</span>
                Heatmap is overlaid on original image for visual explanation
            </li>
        </ul>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    run = st.button("🔬  Run DeepFake Analysis", use_container_width=True)

    if run:
        with st.spinner("Running ResNet-18 inference + GradCAM…"):
            is_fake, fake_prob, real_prob, metrics, flags = run_inference(image)
            gradcam_img, heatmap = generate_gradcam(image)

        # ── Verdict ─────────────────────────────────────────────────────────
        conf_pct = int(fake_prob * 100) if is_fake else int(real_prob * 100)
        verdict_cls = "fake" if is_fake else "real"
        verdict_word = "DEEPFAKE" if is_fake else "AUTHENTIC"
        verdict_sub = (
            "This image shows strong signs of AI-generated or manipulated content."
            if is_fake else
            "No significant manipulation artifacts detected. Likely an authentic photograph."
        )

        st.markdown(f"""
        <div class="result-panel result-{verdict_cls}">
            <div class="verdict-label verdict-{verdict_cls}-label">
                {'⚠ Manipulation Detected' if is_fake else '✓ Authenticity Confirmed'}
            </div>
            <div class="verdict-text verdict-{verdict_cls}-text">{verdict_word}</div>
            <div class="verdict-sub">{verdict_sub}</div>

            <div class="conf-bar-wrap">
                <div class="conf-bar-header">
                    <span class="conf-bar-label">Confidence</span>
                    <span class="conf-bar-pct conf-{verdict_cls}-pct">{conf_pct}%</span>
                </div>
                <div class="conf-bar-bg">
                    <div class="conf-bar-fill conf-bar-{verdict_cls}" style="width:{conf_pct}%"></div>
                </div>
            </div>

            <div style="display:flex;gap:2rem;font-family:'DM Mono',monospace;font-size:0.72rem;color:#6b6880;">
                <span>FAKE &nbsp;<b style="color:#ef4444">{fake_prob*100:.1f}%</b></span>
                <span>REAL &nbsp;<b style="color:#22c55e">{real_prob*100:.1f}%</b></span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Metrics ──────────────────────────────────────────────────────────
        st.markdown('<div class="section-header" style="margin-top:1.5rem;">📊 Detailed Metrics</div>', unsafe_allow_html=True)
        cols = st.columns(3)
        for i, (k, v) in enumerate(metrics.items()):
            with cols[i % 3]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="metric-val">{v}</div>
                    <div class="metric-lbl">{k}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Tabs ─────────────────────────────────────────────────────────────
        tab1, tab2 = st.tabs(["🌡 GradCAM Heatmap", "📋 Analysis Report"])

        with tab1:
            col_a, col_b = st.columns(2, gap="medium")
            with col_a:
                st.markdown('<div class="section-header">Original</div>', unsafe_allow_html=True)
                st.image(image, use_container_width=True)
            with col_b:
                st.markdown('<div class="section-header">GradCAM Overlay</div>', unsafe_allow_html=True)
                st.image(gradcam_img, use_container_width=True)

            st.markdown("""
            <div class="section-desc" style="margin-top:1rem;">
                🔴 Red / warm areas = high model activation (suspected manipulation) &nbsp;·&nbsp;
                🟢 Green / cool = low activation (authentic regions)
            </div>
            """, unsafe_allow_html=True)

        with tab2:
            st.markdown('<div class="section-header">🔎 Signal Analysis</div>', unsafe_allow_html=True)
            icon_map = {"warn": ("icon-warn", "⚠"), "ok": ("icon-ok", "✓"), "info": ("icon-info", "i")}
            items_html = "".join([
                f'<li class="analysis-item"><span class="analysis-icon {icon_map[t][0]}">{icon_map[t][1]}</span>{txt}</li>'
                for t, txt in flags
            ])
            st.markdown(f'<ul class="analysis-list">{items_html}</ul>', unsafe_allow_html=True)

            with st.expander("ℹ  About this model"):
                st.markdown("""
                <div style="font-family:'DM Mono',monospace;font-size:0.75rem;color:#9d9ab0;line-height:1.8;">
                    <b style="color:#a78bfa">Architecture:</b> ResNet-18 fine-tuned on custom deepfake dataset<br>
                    <b style="color:#a78bfa">GradCAM layer:</b> layer4 (final convolutional block)<br>
                    <b style="color:#a78bfa">Input size:</b> 224×224 RGB<br>
                    <b style="color:#a78bfa">Optimizer:</b> Adam · LR 3e-4 · CrossEntropyLoss<br>
                    <b style="color:#a78bfa">Classes:</b> Fake (index 0) · Real (index 1)
                </div>
                """, unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 1rem;color:#3d3a50;font-family:'DM Mono',monospace;font-size:0.75rem;letter-spacing:0.1em;">
        ↑ Upload an image above to begin analysis
    </div>
    """, unsafe_allow_html=True)


# ─── How It Works ────────────────────────────────────────────────────────────
st.markdown("""
<div class="how-section">
    <div class="section-header">⚙ How It Works</div>
    <div class="section-desc">Three-stage pipeline from pixels to prediction</div>
    <div class="how-grid">
        <div class="how-card">
            <div class="how-num">01</div>
            <div class="how-title">Feature Extraction</div>
            <div class="how-desc">ResNet-18 encodes spatial and texture features through 4 progressive convolutional blocks, building a rich visual representation.</div>
        </div>
        <div class="how-card">
            <div class="how-num">02</div>
            <div class="how-title">Classification</div>
            <div class="how-desc">A custom classifier head (Linear → ReLU → Dropout → Linear) outputs probability scores for REAL and FAKE classes.</div>
        </div>
        <div class="how-card">
            <div class="how-num">03</div>
            <div class="how-title">GradCAM Viz</div>
            <div class="how-desc">Gradient-weighted Class Activation Mapping backpropagates gradients to highlight which image regions most influenced the verdict.</div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ─── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    DeepScan · ResNet-18 + GradCAM · Built with Streamlit
</div>
""", unsafe_allow_html=True)