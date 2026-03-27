import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import textwrap
import logging
from io import BytesIO
import base64
import pandas as pd
import os
import time

# =========================================
# 🔧 CONFIGURATION
# =========================================
main_path_of_project = "G:\\GUI Application\\"

logging.basicConfig(filename='predictions.log', level=logging.INFO,
                    format='%(asctime)s - %(message)s')

class_labels = [
    'Apple Apple scab', 'Apple Black rot', 'Apple Cedar apple rust', 'Apple healthy',
    'Bacterial leaf blight in rice leaf', 'Blight in corn Leaf', 'Blueberry healthy',
    'Brown spot in rice leaf', 'Cercospora leaf spot',
    'Cherry (including sour) Powdery mildew', 'Cherry (including_sour) healthy',
    'Common Rust in corn Leaf', 'Corn (maize) healthy', 'Garlic',
    'Grape Black rot', 'Grape Esca Black Measles', 'Grape Leaf blight Isariopsis Leaf Spot',
    'Grape healthy', 'Gray Leaf Spot in corn Leaf', 'Leaf smut in rice leaf',
    'Nitrogen deficiency in plant', 'Orange Haunglongbing Citrus greening', 'Peach healthy',
    'Pepper bell Bacterial spot', 'Pepper bell healthy', 'Potato Early blight',
    'Potato Late blight', 'Potato healthy', 'Raspberry healthy', 'Sogatella rice',
    'Soybean healthy', 'Strawberry Leaf scorch', 'Strawberry healthy',
    'Tomato Bacterial spot', 'Tomato Early blight', 'Tomato Late blight',
    'Tomato Leaf Mold', 'Tomato Septoria leaf spot',
    'Tomato Spider mites Two spotted spider mite', 'Tomato Target Spot',
    'Tomato Tomato mosaic virus', 'Tomato healthy', 'Waterlogging in plant',
    'algal leaf in tea', 'anthracnose in tea', 'bird eye spot in tea',
    'brown blight in tea', 'cabbage looper', 'corn crop', 'ginger', 'healthy tea leaf',
    'lemon canker', 'onion', 'potassium deficiency in plant', 'potato crop',
    'potato hollow heart', 'red leaf spot in tea', 'tomato canker'
]

keras_model_path    = main_path_of_project + "Plaintify_diseases_classifier_model.keras"
json_file_path      = main_path_of_project + "dis.json"
data_info_json_path = main_path_of_project + "datainfo.json"


# =========================================
# 💅 CUSTOM CSS
# =========================================
def inject_css():
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;1,300;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

    /* ─── ROOT & RESET ─── */
    :root {
        --bg:        #07110a;
        --surface:   #0e1f12;
        --card:      #112116;
        --border:    rgba(78,160,90,0.20);
        --green:     #4ea05a;
        --green-d:   #2e7d3c;
        --gold:      #c8a84b;
        --gold-l:    #e8c96e;
        --cream:     #f0ead8;
        --muted:     #7a9e82;
        --danger:    #d96c5c;
        --radius:    14px;
    }

    html, body, [class*="css"] {
        font-family: 'DM Sans', sans-serif !important;
        font-weight: 300;
    }

    /* ─── GLOBAL BACKGROUND ─── */
    .stApp {
        background: var(--bg) !important;
        background-image:
            radial-gradient(ellipse 70% 50% at 10% 0%,  rgba(46,125,60,0.22) 0%, transparent 60%),
            radial-gradient(ellipse 50% 40% at 90% 100%, rgba(200,168,75,0.10) 0%, transparent 55%)
            !important;
    }

    /* ─── HIDE STREAMLIT CHROME ─── */
    #MainMenu, footer, header { visibility: hidden; }
    .block-container {
        max-width: 820px !important;
        padding-top: 2rem !important;
        padding-bottom: 5rem !important;
    }

    /* ─── HERO HEADER ─── */
    .hero {
        text-align: center;
        padding: 48px 0 36px;
        animation: fadeUp 0.8s ease both;
    }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 8px;
        padding: 5px 18px;
        border: 1px solid var(--border);
        border-radius: 999px;
        background: rgba(78,160,90,0.08);
        font-size: 10px; letter-spacing: 2.5px; text-transform: uppercase;
        color: var(--green); margin-bottom: 20px;
    }
    .hero h1 {
        font-family: 'Cormorant Garamond', serif !important;
        font-size: clamp(2.6rem, 6vw, 4.2rem);
        font-weight: 300; line-height: 1.08; letter-spacing: -0.02em;
        color: var(--cream);
    }
    .hero h1 em { font-style: italic; color: var(--gold-l); }
    .hero p {
        margin-top: 16px; color: var(--muted);
        font-size: 0.92rem; line-height: 1.75;
        max-width: 460px; margin-inline: auto;
    }
    .leaf-strip { font-size: 1.4rem; margin-top: 22px; letter-spacing: 8px; }

    /* ─── DIVIDER ─── */
    .fancy-hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border), transparent);
        margin: 10px 0 32px;
    }

    /* ─── UPLOAD ZONE ─── */
    .upload-card {
        background: var(--card);
        border: 1.5px dashed var(--border);
        border-radius: var(--radius);
        padding: 36px 28px;
        text-align: center;
        position: relative;
        overflow: hidden;
        animation: fadeUp 0.9s 0.15s ease both;
    }
    .upload-card::before {
        content: '';
        position: absolute; inset: 0;
        background: radial-gradient(ellipse 60% 60% at 50% 0%, rgba(78,160,90,0.07), transparent);
        pointer-events: none;
    }
    .upload-icon { font-size: 2.6rem; margin-bottom: 12px; }
    .upload-card h3 {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.5rem; font-weight: 400;
        color: var(--cream); margin-bottom: 6px;
    }
    .upload-card p { color: var(--muted); font-size: 0.85rem; }

    /* ─── STREAMLIT FILE UPLOADER OVERRIDE ─── */
    [data-testid="stFileUploader"] {
        background: transparent !important;
    }
    [data-testid="stFileUploader"] > div {
        background: rgba(14,31,18,0.6) !important;
        border: 1.5px dashed rgba(78,160,90,0.30) !important;
        border-radius: var(--radius) !important;
        padding: 24px !important;
    }
    [data-testid="stFileUploader"] label { color: var(--cream) !important; }
    [data-testid="stFileUploaderDropzoneInstructions"] {
        color: var(--muted) !important;
    }
    [data-testid="stFileUploaderDropzoneInstructions"] svg { fill: var(--green) !important; }
    [data-testid="stFileUploader"] button {
        background: var(--green-d) !important;
        color: var(--cream) !important;
        border: none !important;
        border-radius: 8px !important;
    }

    /* ─── IMAGE PREVIEW CARD ─── */
    .img-preview-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 24px;
        text-align: center;
        animation: fadeUp 0.6s ease both;
    }
    .img-preview-card img {
        border-radius: 10px;
        border: 2px solid rgba(78,160,90,0.25);
        box-shadow: 0 8px 32px rgba(0,0,0,0.4);
    }
    .img-label {
        margin-top: 10px;
        font-size: 0.78rem;
        color: var(--muted);
        letter-spacing: 1.5px;
        text-transform: uppercase;
    }

    /* ─── RESULT CARD ─── */
    .result-card {
        background: linear-gradient(135deg, rgba(46,125,60,0.12), rgba(78,160,90,0.06));
        border: 1px solid rgba(78,160,90,0.35);
        border-radius: var(--radius);
        padding: 28px 32px;
        display: flex; align-items: center; gap: 20px;
        animation: popIn 0.5s cubic-bezier(0.34,1.56,0.64,1) both;
    }
    .result-icon { font-size: 2.4rem; flex-shrink: 0; }
    .result-label {
        font-size: 0.72rem; letter-spacing: 2px; text-transform: uppercase;
        color: var(--green); margin-bottom: 4px;
    }
    .result-name {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.9rem; font-weight: 400; color: var(--cream);
        line-height: 1.2;
    }
    .confidence-pill {
        display: inline-flex; align-items: center; gap: 6px;
        margin-top: 10px;
        padding: 4px 14px;
        background: rgba(200,168,75,0.12);
        border: 1px solid rgba(200,168,75,0.30);
        border-radius: 999px;
        font-size: 0.82rem; color: var(--gold-l);
    }
    .timer-pill {
        display: inline-flex; align-items: center; gap: 6px;
        margin-top: 6px;
        padding: 4px 14px;
        background: rgba(78,160,90,0.10);
        border: 1px solid rgba(78,160,90,0.28);
        border-radius: 999px;
        font-size: 0.82rem; color: #a8d5b0;
    }

    /* ─── CONFIDENCE BAR ─── */
    .conf-bar-wrap {
        background: rgba(255,255,255,0.05);
        border-radius: 999px; height: 6px; overflow: hidden;
        margin-top: 12px;
    }
    .conf-bar-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, var(--green-d), var(--gold));
        transition: width 1s ease;
    }

    /* ─── INFO SECTION CARDS ─── */
    .section-heading {
        font-family: 'Cormorant Garamond', serif;
        font-size: 1.35rem; font-weight: 400;
        color: var(--cream);
        border-left: 3px solid var(--green);
        padding-left: 14px;
        margin: 32px 0 16px;
    }
    .info-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 16px;
    }
    .info-chip {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 10px;
        padding: 14px 18px;
    }
    .info-chip-label {
        font-size: 0.68rem; letter-spacing: 2px; text-transform: uppercase;
        color: var(--muted); margin-bottom: 4px;
    }
    .info-chip-value {
        font-size: 0.92rem; color: var(--cream); font-weight: 400;
    }
    .cure-box {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 22px 24px;
        font-size: 0.88rem;
        line-height: 1.8;
        color: #c4d9c8;
    }

    /* ─── WARNING / NO-DATA ─── */
    .warn-box {
        background: rgba(217,108,92,0.08);
        border: 1px solid rgba(217,108,92,0.25);
        border-radius: 10px;
        padding: 14px 20px;
        font-size: 0.86rem;
        color: #e8a49a;
        display: flex; align-items: center; gap: 10px;
    }

    /* ─── SPINNER OVERRIDE ─── */
    [data-testid="stSpinner"] { color: var(--green) !important; }

    /* ─── FOOTER ─── */
    .app-footer {
        text-align: center;
        margin-top: 56px;
        padding-top: 24px;
        border-top: 1px solid var(--border);
        font-size: 0.78rem;
        color: var(--muted);
        letter-spacing: 0.5px;
    }
    .app-footer span { color: var(--green); }

    /* ─── ANIMATIONS ─── */
    @keyframes fadeUp {
        from { opacity: 0; transform: translateY(18px); }
        to   { opacity: 1; transform: translateY(0); }
    }
    @keyframes popIn {
        from { opacity: 0; transform: scale(0.92); }
        to   { opacity: 1; transform: scale(1); }
    }
    </style>
    """, unsafe_allow_html=True)


# =========================================
# 🧠 MODEL & DATA LOADERS
# =========================================
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model(keras_model_path)
        return model
    except Exception as e:
        return None


@st.cache_data
def load_dis_json():
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


@st.cache_data
def load_data_info():
    try:
        with open(data_info_json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return pd.DataFrame(data["data"])
    except Exception:
        return pd.DataFrame()


# =========================================
# 🖼 IMAGE HELPERS
# =========================================
def preprocess_image(img: Image.Image):
    img = img.resize((224, 224), Image.Resampling.LANCZOS)
    arr = np.array(img, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


def classify_image_keras(model, img_array):
    preds = model.predict(img_array, verbose=0)
    idx = np.argmax(preds[0])
    return class_labels[idx], float(preds[0][idx])


def img_to_b64(img: Image.Image) -> str:
    img = img.resize((260, 260), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def is_healthy(name: str) -> bool:
    return "healthy" in name.lower()


# =========================================
# 🌿 MAIN APP
# =========================================
def main():
    st.set_page_config(
        page_title="Plaintify — Leaf Disease Classifier",
        page_icon="🌿",
        layout="centered",
        initial_sidebar_state="collapsed",
    )

    inject_css()

    # ── HERO ──────────────────────────────────────────────
    st.markdown("""
    <div class="hero">
        <div class="hero-badge">🌿 &nbsp; AI-Powered Diagnostics</div>
        <h1><em>Plaintify</em><br>Leaf Disease Classifier</h1>
        <p>Upload a leaf photograph and our model will identify the disease,
           confidence level, and recommended treatment.</p>
        <div class="leaf-strip">🍃 🌱 🍀 🌿 🌾</div>
    </div>
    <hr class="fancy-hr"/>
    """, unsafe_allow_html=True)

    # ── LOAD RESOURCES ────────────────────────────────────
    model        = load_model()
    dis_DB_cure  = load_dis_json()
    data_info_df = load_data_info()

    if model is None:
        st.markdown("""
        <div class="warn-box">
            ⚠️ &nbsp; <strong>Model not found.</strong>
            Place <code>Plaintify_diseases_classifier_model.keras</code>
            in the project folder and restart.
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── UPLOAD ────────────────────────────────────────────
    st.markdown('<div class="section-heading">📤 Upload a Leaf Image</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Drop your JPG / PNG here",
        type=["jpg", "jpeg", "png"],
        label_visibility="collapsed",
    )

    # ── PROCESS ───────────────────────────────────────────
    if uploaded_file:
        try:
            image = Image.open(uploaded_file).convert("RGB")
        except Exception:
            st.markdown('<div class="warn-box">❌ &nbsp; Could not open image. Please upload a valid JPG or PNG.</div>', unsafe_allow_html=True)
            st.stop()

        col_img, col_info = st.columns([1, 1.6], gap="large")

        with col_img:
            st.markdown(f"""
            <div class="img-preview-card">
                <img src="data:image/png;base64,{img_to_b64(image)}" width="220" height="220"/>
                <div class="img-label">📷 &nbsp; Uploaded image</div>
            </div>
            """, unsafe_allow_html=True)

        with col_info:
            with st.spinner("🔬 Analysing leaf…"):
                img_array = preprocess_image(image)
                t_start = time.perf_counter()
                predicted_class, confidence = classify_image_keras(model, img_array)
                t_elapsed = time.perf_counter() - t_start

            logging.info(f"Prediction: {predicted_class} | Confidence: {confidence:.4f} | Time: {t_elapsed:.6f}s | File: {uploaded_file.name}")

            icon     = "✅" if is_healthy(predicted_class) else "🦠"
            conf_pct = confidence * 100
            bar_color = "#4ea05a" if conf_pct >= 70 else "#c8a84b" if conf_pct >= 40 else "#d96c5c"

            st.markdown(f"""
            <div class="result-card">
                <div class="result-icon">{icon}</div>
                <div>
                    <div class="result-label">Detected Condition</div>
                    <div class="result-name">{predicted_class.upper()}</div>
                    <div class="confidence-pill">
                        🧠 &nbsp; Confidence &nbsp;·&nbsp; <strong>{conf_pct:.1f}%</strong>
                    </div>
                    <div class="timer-pill">
                        ⏱ &nbsp; Classified in &nbsp;·&nbsp; <strong>{t_elapsed:.6f}s</strong>
                    </div>
                    <div class="conf-bar-wrap">
                        <div class="conf-bar-fill"
                             style="width:{conf_pct:.1f}%;background:linear-gradient(90deg,{bar_color},{bar_color}aa);">
                        </div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── DATA INFO JSON (directly below Detected Condition) ────────
            st.markdown('<div class="section-heading" style="margin-top:24px;">📘 Disease Information</div>', unsafe_allow_html=True)

            disease_name = predicted_class.lower()
            matched = pd.DataFrame()
            if not data_info_df.empty:
                matched = data_info_df[data_info_df["name"].str.contains(disease_name, case=False, na=False)]

            if not matched.empty:
                info = matched.iloc[0]
                plant_val  = info.get("plantName", "—")
                part_val   = info.get("plantPart", "—")
                dtype_val  = info.get("diseaseType", "—")
                cure_val   = info.get("cure", "No cure information available.")

                st.markdown(f"""
                <div class="info-grid">
                    <div class="info-chip">
                        <div class="info-chip-label">🌿 Plant</div>
                        <div class="info-chip-value">{plant_val.upper()}</div>
                    </div>
                    <div class="info-chip">
                        <div class="info-chip-label">🍃 Part Affected</div>
                        <div class="info-chip-value">{part_val.upper()}</div>
                    </div>
                    <div class="info-chip" style="grid-column:1/-1;">
                        <div class="info-chip-label">🧫 Disease Type</div>
                        <div class="info-chip-value">{dtype_val.upper()}</div>
                    </div>
                </div>
                <div class="cure-box">
                    <strong style="color:var(--gold-l);font-size:0.78rem;letter-spacing:1.5px;text-transform:uppercase;">
                        Recommended Treatment
                    </strong><br/><br/>
                    {cure_val.upper()}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-box">⚠️ &nbsp; No matching entry found in <code>datainfo.json</code>.</div>', unsafe_allow_html=True)

        st.markdown('<hr class="fancy-hr" style="margin-top:28px;"/>', unsafe_allow_html=True)

        # ── DIS JSON ───────────────────────────────────────
        st.markdown('<div class="section-heading">💊 Additional Cure Reference</div>', unsafe_allow_html=True)

        cure_text = None
        if disease_name in dis_DB_cure:
            cure_text = dis_DB_cure[disease_name]
        else:
            for key, val in dis_DB_cure.items():
                if disease_name in key.lower():
                    cure_text = val
                    break

        if cure_text:
            st.markdown(f'<div class="cure-box">{cure_text}</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="warn-box">⚠️ &nbsp; No cure data found in <code>dis.json</code>.</div>', unsafe_allow_html=True)

    # ── FOOTER ────────────────────────────────────────────
    st.markdown("""
    <div class="app-footer">
        Made with <span>♥</span> &nbsp;·&nbsp;
        <strong>Plaintify Model last update:23 Nov 2025</strong> &nbsp;·&nbsp;
        Streamlit GUI by <em>Sadman Sakib Mahi</em>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()