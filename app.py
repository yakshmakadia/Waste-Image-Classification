import streamlit as st

import random

import time

import base64

from pathlib import Path

import torch

import torchvision.models as models

import torch.nn as nn

from PIL import Image

import torchvision.transforms as transforms
 
class ResNet(nn.Module):

    def __init__(self):

        super().__init__()
 
        self.network = models.resnet50(weights=None)
 
        num_ftrs = self.network.fc.in_features
 
        self.network.fc = nn.Linear(

            num_ftrs,

            6

        )
 
    def forward(self, xb):

        return torch.sigmoid(self.network(xb))
 
CLASS_NAMES = [

    "Cardboard",

    "Glass",

    "Metal",

    "Paper",

    "Plastic",

    "Trash"

]
 
model = ResNet()
 
model.load_state_dict(

    torch.load(

        "waste_classifier.pth",

        map_location=torch.device("cpu")

    )

)
 
model.eval()
 
transformations = transforms.Compose([

    transforms.Resize((256, 256)),

    transforms.ToTensor()

])
 
def predict_image(uploaded_file):
 
    image = Image.open(uploaded_file).convert("RGB")
 
    img_tensor = transformations(image)
 
    img_tensor = img_tensor.unsqueeze(0)
 
    with torch.no_grad():
 
        output = model(img_tensor)
 
        probabilities = torch.softmax(output, dim=1)
 
        confidence, pred = torch.max(probabilities, 1)
 
        predicted_class = CLASS_NAMES[pred.item()]
 
        confidence_score = round(

            float(confidence.item() * 100),

            2

        )
 
    return predicted_class, confidence_score
 
# ----------------------------

# BACKGROUND IMAGE (base64 encode local file)

# ----------------------------

def get_base64(file_path):

    with open(file_path, "rb") as f:

        return base64.b64encode(f.read()).decode()
 
BG_PATH = Path(__file__).parent / "background.png"

bg_base64 = get_base64(BG_PATH) if BG_PATH.exists() else ""
 
# ----------------------------

# PAGE CONFIG

# ----------------------------

st.set_page_config(

    page_title="Waste Image Classification",

    page_icon="♻️",

    layout="wide",

    initial_sidebar_state="collapsed",

)
 
# ----------------------------

# CUSTOM CSS

# ----------------------------

st.markdown("""
<style>

/* Prevent padding/borders from ever pushing content past its box —

   this is what was letting the category pills spill outside the card. */

*, *::before, *::after {

    box-sizing: border-box;

}
 
/* ---------- Global ---------- */

html, body, [class*="css"]  {

    font-family: 'Segoe UI', 'Poppins', sans-serif;

}

.stApp {

    background-color: #060d08;

}
 
/* Hide default streamlit chrome */

#MainMenu, header, footer {visibility: hidden;}
 
/* Match the content width to the background image (1536px wide)

   so the layout fills the screen the same way the background does. */

.block-container {

    max-width: 1536px;

    padding-top: 1.5rem;

    padding-bottom: 3rem;

    padding-left: 2.5rem;

    padding-right: 2.5rem;

    margin: 0 auto;

}
 
/* ---------- Segmented control (nav buttons) ---------- */

[data-testid="stSegmentedControl"] {

    margin-top: 0;

}

[data-testid="stSegmentedControl"] button {

    background: #12241a !important;

    color: white !important;

    border: 1px solid #2f5f3a !important;

    border-radius: 14px !important;

    padding: 10px 18px !important;

    font-weight: 600 !important;

    font-size: 15px !important;

}

[data-testid="stSegmentedControl"] button[data-selected="true"] {

    background: #6fd66f !important;

    color: #0d1b14 !important;

    font-weight: 700 !important;

    border-color: #6fd66f !important;

}
 
/* ---------- Navbar box (real container, so it actually wraps its content) ---------- */

.st-key-navbar_box {

    background: rgba(13,27,20,0.92);

    border: 1px solid rgba(111,214,111,0.25);

    border-radius: 18px;

    padding: 16px 24px;

    margin-bottom: 28px;

}

.navbar-brand {

    display: flex;

    align-items: center;

    gap: 14px;

    height: 100%;

    transform: translateY(-5px);

}

.navbar-brand-icon {

    font-size: 32px;

}

.navbar-brand-title {

    color: white;

    font-size: 25px;

    font-weight: 800;

}

.navbar-brand-title span {

    color: #6fd66f;

}

.navbar-brand-sub {

    color: #9db3a4;

    font-size: 14px;

    margin-top: 3px;

    position: relative;

    top: -10px;

}
 
/* ---------- Card panels (real containers) ---------- */

.st-key-upload_panel, .st-key-preview_panel, .st-key-result_panel {

    background-color: #12241a;

    border-radius: 16px;

    padding: 30px;

    color: white;

    height: 100%;

}

.panel-header {

    margin-bottom: 24px;

}

.panel-title {

    display: flex;

    align-items: center;

    gap: 10px;

    font-size: 21px;

    font-weight: 700;

    margin-bottom: 6px;

}

.panel-subtitle {

    color: #9db3a4;

    font-size: 15px;

    margin-left: 36px;

}

.step-badge {

    background-color: #4caf50;

    color: white;

    width: 28px;

    height: 28px;

    border-radius: 50%;

    display: inline-flex;

    align-items: center;

    justify-content: center;

    font-size: 15px;

    font-weight: 700;

    flex-shrink: 0;

}
 
/* Empty preview placeholder */

.preview-empty {

    border: 2px dashed #3a5f45;

    border-radius: 14px;

    min-height: 300px;

    display: flex;

    flex-direction: column;

    align-items: center;

    justify-content: center;

    gap: 12px;

    background: linear-gradient(135deg, rgba(18,36,26,0.4), rgba(10,20,14,0.4));

}

.preview-empty-icon {

    font-size: 40px;

    color: #5f7a68;

    opacity: 0.85;

}

.preview-empty-text {

    color: #7f9488;

    font-size: 15px;

}
 
.upload-caption {

    text-align: center;

    color: #9db3a4;

    font-size: 14px;

    margin-top: 18px;

}
 
/* Predict button – centered, auto width, roomy */

div.stButton {

    text-align: center;

    margin-top: 24px;

}

div.stButton > button {

    background: linear-gradient(90deg, #3ea55e, #6fd66f);

    color: #0d1b14;

    font-weight: 700;

    border: none;

    border-radius: 12px;

    padding: 15px 32px;

    width: auto;

    min-width: 300px;

    display: inline-block;

    min-height: 56px;

    font-size: 18px;

    white-space: nowrap;

    overflow: visible;

    line-height: 1.2;

}

div.stButton > button p {

    white-space: nowrap;

    font-size: 18px;

    overflow: visible;

}

div.stButton > button:hover {

    background: linear-gradient(90deg, #6fd66f, #3ea55e);

    color: #0d1b14;

}
 
/* Result section */

.result-empty-box {

    display: flex;

    align-items: center;

    justify-content: center;

    gap: 14px;

    border: 1px solid #234a2e;

    border-radius: 14px;

    padding: 24px 26px;

    background: rgba(255,255,255,0.02);

    color: #cfcfcf;

    font-size: 15px;

}

.result-empty-icon {

    font-size: 22px;

    color: #6fd66f;

}

.result-circle-wrap {

    text-align: center;

}

.result-circle {

    width: 190px;

    height: 190px;

    border-radius: 50%;

    border: 3px solid #6fd66f;

    box-shadow: 0 0 26px rgba(111,214,111,0.45);

    display: flex;

    align-items: center;

    justify-content: center;

    margin: 0 auto 16px auto;

    font-size: 62px;

}

.predicted-label {

    color: #cfcfcf;

    font-size: 15px;

    margin-bottom: 3px;

}

.predicted-category {

    color: #6fd66f;

    font-size: 38px;

    font-weight: 800;

    margin-bottom: 12px;

}

.recyclable-badge {

    background-color: #234a2e;

    color: #6fd66f;

    display: inline-block;

    padding: 8px 20px;

    border-radius: 20px;

    font-weight: 600;

    font-size: 15px;

    margin-bottom: 14px;

}

.result-desc {

    color: #cfcfcf;

    font-size: 14px;

}

.confidence-title {

    font-size: 16px;

    font-weight: 600;

    margin-bottom: 8px;

}

.confidence-score {

    color: #6fd66f;

    font-size: 38px;

    font-weight: 800;

    margin-bottom: 14px;

}

.progress-track {

    background-color: #2a3d31;

    border-radius: 10px;

    height: 12px;

    width: 100%;

    margin-bottom: 8px;

    overflow: hidden;

}

.progress-fill {

    background: linear-gradient(90deg, #3ea55e, #6fd66f);

    height: 100%;

    border-radius: 10px;

}
 
/* ---------- Feature strip (dark themed) ---------- */

.feature-card {

    background-color: #12241a;

    border: 1px solid rgba(111,214,111,0.15);

    border-radius: 16px;

    padding: 22px;

    display: flex;

    gap: 14px;

    align-items: flex-start;

    height: 100%;

}

.feature-icon-badge {

    width: 46px;

    height: 46px;

    border-radius: 50%;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 22px;

    flex-shrink: 0;

}

.feature-title {

    font-weight: 700;

    color: #ffffff;

    font-size: 17px;

    margin-bottom: 3px;

}

.feature-desc {

    color: #9db3a4;

    font-size: 14px;

    line-height: 1.4;

}

.footer-text {

    text-align: center;

    color: #7f9488;

    font-size: 14px;

    margin-top: 24px;

}
 
/* ---------- Native Streamlit file uploader, restyled ---------- */

[data-testid="stFileUploader"] {

    background: linear-gradient(135deg, rgba(18,36,26,0.55), rgba(10,20,14,0.55));

    border: 2px dashed #3a5f45;

    border-radius: 14px;

    padding: 30px;

}

[data-testid="stFileUploader"]:hover {

    border-color: #6fd66f;

}

[data-testid="stFileUploaderDropzone"] {

    border: none !important;

    background: transparent !important;

    min-height: 220px;

    flex-direction: column !important;

    align-items: center !important;

    justify-content: center !important;

    gap: 10px !important;

}

[data-testid="stFileUploaderDropzoneInstructions"] {

    flex-direction: column !important;

    align-items: center !important;

    gap: 2px !important;

}

[data-testid="stFileUploaderDropzoneInstructions"] svg {

    width: 44px !important;

    height: 44px !important;

    fill: #6fd66f !important;

    color: #6fd66f !important;

    margin-bottom: 6px;

}

[data-testid="stFileUploaderDropzoneInstructions"] span {

    font-size: 0 !important;

}

[data-testid="stFileUploaderDropzoneInstructions"] span::before {

    content: "Drag & Drop your image here";

    font-size: 18px;

    font-weight: 700;

    color: #ffffff;

    display: block;

}

[data-testid="stFileUploaderDropzoneInstructions"] small {

    font-size: 0 !important;

}

[data-testid="stFileUploaderDropzoneInstructions"] small::before {

    content: "or browse files";

    font-size: 14px;

    color: #6fd66f;

    text-decoration: underline;

    display: block;

    margin-top: 4px;

}

/* NOTE: the real "Browse files" button keeps its own default label here

   (no ::before override) — adding a second custom label on top of it is

   what caused the "Upload Upload" duplicate text before. */

[data-testid="stFileUploaderDropzone"] button {

    background: linear-gradient(90deg, #3ea55e, #6fd66f) !important;

    color: #0d1b14 !important;

    border: none !important;

    border-radius: 30px !important;

    padding: 12px 26px !important;

    font-weight: 700 !important;

    font-size: 15px !important;

    margin-top: 8px !important;

}
 
/* ---------- How It Works page ---------- */

.how-hero {

    text-align: center;

    margin: 10px 0 44px 0;

}

.how-hero-title {

    font-size: 42px;

    font-weight: 800;

    color: white;

}

.how-hero-title .highlight {

    color: #6fd66f;

}

.how-hero-sub {

    color: #9db3a4;

    font-size: 16px;

    margin-top: 10px;

}

.step-card {

    background-color: #12241a;

    border: 1px solid rgba(111,214,111,0.2);

    border-radius: 16px;

    padding: 36px 18px 26px 18px;

    text-align: center;

    position: relative;

    height: 100%;

    width: 100%;

    color: white;

    overflow: hidden;

    margin-top: 25px;

}

.step-number {

    position: absolute;

    top: 20px;

    left: 50%;

    transform: translateX(-50%);

    width: 36px;

    height: 36px;

    border-radius: 50%;

    background: #4caf50;

    color: white;

    display: flex;

    align-items: center;

    justify-content: center;

    font-weight: 700;

    font-size: 16px;

    border: 3px solid #0d1b14;

}

.step-icon {

    font-size: 36px;

    margin-bottom: 16px;

    margin-top: 35px;

}

.step-title {

    font-size: 19px;

    font-weight: 700;

    margin-bottom: 10px;

    color: white;

}

.step-desc {

    font-size: 14px;

    color: #9db3a4;

    line-height: 1.55;

}

.step-pill {

    display: inline-block;

    margin-top: 18px;

    background: #1a2e21;

    color: #cfcfcf;

    font-size: 13px;

    padding: 7px 16px;

    border-radius: 20px;

    border: 1px solid #2f5f3a;

}

.category-grid {

    display: grid;

    grid-template-columns: repeat(2, minmax(0, 1fr));

    gap: 8px;

    margin-top: 18px;

    width: 100%;

}

.category-pill {

    display: flex;

    align-items: center;

    gap: 5px;

    background: #1a2e21;

    border-radius: 10px;

    padding: 9px 10px;

    font-size: 15px;

    color: #e5e5e5;

    min-width: 0;

    overflow: hidden;

}

.category-pill-label {

    overflow: hidden;

    text-overflow: ellipsis;

    white-space: nowrap;

}

.category-pill-icon {

    width: 20px;

    height: 20px;

    border-radius: 5px;

    display: flex;

    align-items: center;

    justify-content: center;

    font-size: 11px;

    flex-shrink: 0;

}

.step-arrow {

    display: flex;

    align-items: center;

    justify-content: center;

    height: 100%;

    padding-top: 55px;

}

.how-footer-box {

    display: flex;

    align-items: center;

    gap: 16px;

    background-color: #12241a;

    border: 1px solid rgba(111,214,111,0.2);

    border-radius: 16px;

    padding: 26px 28px;

    margin-top: 26px;

    justify-content: center;

}

.how-footer-icon {

    font-size: 30px;

}

.how-footer-title {

    font-size: 18px;

    font-weight: 700;

    color: white;

}

.how-footer-sub {

    font-size: 14px;

    color: #9db3a4;

    margin-top: 3px;

}
 
/* ---------- Mobile: stack columns, shrink big titles ---------- */

@media (max-width: 900px) {

    [data-testid="stHorizontalBlock"] {

        flex-wrap: wrap !important;

    }

    [data-testid="stHorizontalBlock"] > div {

        width: 100% !important;

        flex: 1 1 100% !important;

        margin-bottom: 16px;

    }

    .navbar-brand-title { font-size: 20px; }

    .how-hero-title { font-size: 30px; }

    div.stButton > button { min-width: 220px; }

}
</style>

""", unsafe_allow_html=True)
 
# Inject the leafy background image

st.markdown(f"""
<style>

.stApp {{

    background-image: url("data:image/png;base64,{bg_base64}");

    background-size: cover;

    background-position: center;

    background-repeat: no-repeat;

    background-attachment: fixed;

}}
</style>

""", unsafe_allow_html=True)
 
# ----------------------------

# NAVBAR (a real st.container so the border/background actually

# wraps the brand text + nav buttons, instead of floating above them)

# ----------------------------

with st.container(key="navbar_box"):

    nav_col1, nav_col2 = st.columns([7, 3],vertical_alignment="center",gap="small")
 
    with nav_col1:

        st.markdown("""
<div class="navbar-brand">
<div class="navbar-brand-icon">♻️</div>
<div>
<div class="navbar-brand-title">Waste Image <span>Classification</span></div>
<div class="navbar-brand-sub">A cleaner tomorrow starts with smarter choices</div>
</div>
</div>

        """, unsafe_allow_html=True)
 
    with nav_col2:

        page = st.segmented_control(

            "",

            ["🏠 Home", "⚙️ How It Works", "ℹ️ About"],

            default="🏠 Home",

            label_visibility="collapsed",

        )
 
if page == "🏠 Home":
 
    # ----------------------------

    # STEP 1 & 2 : UPLOAD + PREVIEW

    # ----------------------------

    col1, col2 = st.columns(2)
 
    with col1:

        with st.container(key="upload_panel"):

            st.markdown("""
<div class="panel-header">
<div class="panel-title"><span class="step-badge">1</span> Upload Waste Image</div>
<div class="panel-subtitle">Choose an image of waste to classify its type</div>
</div>

            """, unsafe_allow_html=True)
 
            uploaded_file = st.file_uploader(

                "Upload waste image",

                type=["jpg", "jpeg", "png"],

                label_visibility="collapsed",

            )
 
            st.markdown("""
<div class="upload-caption">Supported formats: JPG, PNG | Max size: 20 MB</div>

            """, unsafe_allow_html=True)
 
    with col2:

        with st.container(key="preview_panel"):

            st.markdown("""
<div class="panel-header">
<div class="panel-title"><span class="step-badge">2</span> Image Preview</div>
<div class="panel-subtitle">Preview your uploaded image</div>
</div>

            """, unsafe_allow_html=True)
 
            if uploaded_file is not None:

                st.image(uploaded_file, use_container_width=True)

            else:

                st.markdown("""
<div class="preview-empty">
<div class="preview-empty-icon">🖼️</div>
<div class="preview-empty-text">Uploaded image will appear here</div>
</div>

                """, unsafe_allow_html=True)
 
        predict_clicked = st.button("✨  Predict Waste Type →")
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ----------------------------

    # STEP 3 : PREDICTION RESULT

    # ----------------------------
 
    CATEGORY_INFO = {

        "Paper": {

            "icon": "📄",

            "recyclable": True,

            "desc": "Paper waste can be recycled into new paper products."

        },

        "Glass": {

            "icon": "🍾",

            "recyclable": True,

            "desc": "Glass waste can be recycled endlessly without losing quality."

        },

        "Cardboard": {

            "icon": "📦",

            "recyclable": True,

            "desc": "Cardboard waste can be recycled into new packaging materials."

        },

        "Metal": {

            "icon": "🥫",

            "recyclable": True,

            "desc": "Metal waste can be melted down and reused efficiently."

        },

        "Plastic": {

            "icon": "🧴",

            "recyclable": True,

            "desc": "Plastic waste can be recycled and reused to create new products."

        },

        "Trash": {

            "icon": "🍎",

            "recyclable": False,

            "desc": "Food waste can be composted into natural fertilizer."

        },

    }
 
    if "result" not in st.session_state:

        st.session_state.result = None
 
    if predict_clicked:

        if uploaded_file is None:

            st.warning("Please upload an image first.")

        else:

            with st.spinner("Analyzing image..."):

                time.sleep(1.2)

            category, confidence = predict_image(uploaded_file)

            st.session_state.result = {"category": category, "confidence": confidence}
 
    with st.container(key="result_panel"):

        st.markdown("""
<div class="panel-header">
<div class="panel-title"><span class="step-badge">3</span> Prediction Result</div>
<div class="panel-subtitle">View the classification result and confidence score</div>
</div>

        """, unsafe_allow_html=True)
 
        if st.session_state.result:

            cat = st.session_state.result["category"]

            conf = st.session_state.result["confidence"]

            info = CATEGORY_INFO[cat]

            recyclable_badge = "✅ Recyclable" if info["recyclable"] else "🍃 Compostable"
 
            rcol1 = st.columns(1)[0]
 
            with rcol1:

                st.markdown(f"""
<div class="result-circle-wrap">
<div class="result-circle">{info['icon']}</div>
<div class="predicted-label">Predicted Category</div>
<div class="predicted-category">{cat}</div>
<div class="recyclable-badge">{recyclable_badge}</div>
<div class="result-desc">{info['desc']}</div>
</div>

                """, unsafe_allow_html=True)

        else:

            st.markdown("""
<div class="result-empty-box">
<span class="result-empty-icon">📊</span>
<span>Upload an image and click "Predict Waste Type" to see the result here.</span>
</div>

            """, unsafe_allow_html=True)
 
    st.markdown("<br>", unsafe_allow_html=True)
 
    # ----------------------------

    # FEATURE STRIP

    # ----------------------------

    f1, f2, f3, f4 = st.columns(4)
 
    features = [

        ("🌱", "Eco Friendly", "Helps in proper waste management", "#1f3d28"),

        ("🎯", "High Accuracy", "AI model with high prediction accuracy", "#3a1f33"),

        ("⚡", "Fast & Easy", "Get results in just a few clicks", "#3d3a1f"),

        ("❤️", "Clean Planet", "Small steps today, a greener tomorrow", "#3d1f22"),

    ]
 
    for col, (icon, title, desc, badge_bg) in zip([f1, f2, f3, f4], features):

        with col:

            st.markdown(f"""
<div class="feature-card">
<div class="feature-icon-badge" style="background:{badge_bg};">{icon}</div>
<div>
<div class="feature-title">{title}</div>
<div class="feature-desc">{desc}</div>
</div>
</div>

            """, unsafe_allow_html=True)
 
    st.markdown("""
<div class="footer-text">© 2026 Waste Image Classifier | Built with 💚 for a cleaner planet</div>

    """, unsafe_allow_html=True)
 
elif page == "⚙️ How It Works":
 
    st.markdown("""
<div class="how-hero">
<div class="how-hero-title">⚙️ How It <span class="highlight">Works</span></div>
<div class="how-hero-sub">From image to impact — in just a few simple steps</div>
</div>

    """, unsafe_allow_html=True)
 
    HOW_CATEGORIES = [

        ("Cardboard", "📦", "#5a3c1e"),

        ("Glass", "🍷", "#1c4a63"),

        ("Metal", "🥫", "#3a3f44"),

        ("Paper", "📄", "#16233d"),

        ("Plastic", "🧴", "#5a1c2b"),

        ("Trash", "🗑️", "#2b2b2b"),

    ]

    category_pills_html = "".join(

        f'<div class="category-pill"><span class="category-pill-icon" style="background:{bg};">{icon}</span><span class="category-pill-label">{label}</span></div>'

        for label, icon, bg in HOW_CATEGORIES

    )
 
    c1, a1, c2, a2, c3, a3, c4 = st.columns([3, 0.5, 3, 0.5, 3, 0.5, 3])
 
    with c1:

        st.markdown("""
<div class="step-card">
<div class="step-number">1</div>
<div class="step-icon">📤</div>
<div class="step-title">Upload Waste Image</div>
<div class="step-desc">Upload an image using drag-and-drop or browse files from your device.</div>
<div class="step-pill">Supports: JPG, PNG</div>
</div>

        """, unsafe_allow_html=True)
 
    with a1:

        st.markdown("""
<div class="step-arrow">
<svg width="50" height="40" viewBox="0 0 50 40" xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="ah1" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
<polygon points="0 0, 6 3, 0 6" fill="#6fd66f"></polygon>
</marker>
</defs>
<path d="M2 8 C 20 8, 20 32, 46 32" stroke="#6fd66f" stroke-width="2" stroke-dasharray="4 4" fill="none" marker-end="url(#ah1)"></path>
</svg>
</div>

        """, unsafe_allow_html=True)
 
    with c2:

        st.markdown("""
<div class="step-card">
<div class="step-number">2</div>
<div class="step-icon">🔍</div>
<div class="step-title">Preview Image</div>
<div class="step-desc">The uploaded image is displayed so you can verify it before classification.</div>
</div>

        """, unsafe_allow_html=True)
 
    with a2:

        st.markdown("""
<div class="step-arrow">
<svg width="50" height="40" viewBox="0 0 50 40" xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="ah2" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
<polygon points="0 0, 6 3, 0 6" fill="#6fd66f"></polygon>
</marker>
</defs>
<path d="M2 8 C 20 8, 20 32, 46 32" stroke="#6fd66f" stroke-width="2" stroke-dasharray="4 4" fill="none" marker-end="url(#ah2)"></path>
</svg>
</div>

        """, unsafe_allow_html=True)
 
    with c3:

        st.markdown("""
<div class="step-card">
<div class="step-number">3</div>
<div class="step-icon">🧠</div>
<div class="step-title">AI Analysis</div>
<div class="step-desc">Our trained ResNet50 model analyzes the image and predicts the type of waste.</div>
</div>

        """, unsafe_allow_html=True)
 
    with a3:

        st.markdown("""
<div class="step-arrow">
<svg width="50" height="40" viewBox="0 0 50 40" xmlns="http://www.w3.org/2000/svg">
<defs>
<marker id="ah3" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
<polygon points="0 0, 6 3, 0 6" fill="#6fd66f"></polygon>
</marker>
</defs>
<path d="M2 8 C 20 8, 20 32, 46 32" stroke="#6fd66f" stroke-width="2" stroke-dasharray="4 4" fill="none" marker-end="url(#ah3)"></path>
</svg>
</div>

        """, unsafe_allow_html=True)
 
    with c4:

        st.markdown(f"""
<div class="step-card">
<div class="step-number">4</div>
<div class="step-icon">🗑️</div>
<div class="step-title">Prediction Result</div>
<div class="step-desc">The model classifies the waste into one of six categories and shows the confidence score.</div>
<div class="category-grid">{category_pills_html}</div>
</div>

        """, unsafe_allow_html=True)
 
    st.markdown("""
<div class="how-footer-box">
<div class="how-footer-icon">🌱</div>
<div>
<div class="how-footer-title">Small steps. A cleaner tomorrow.</div>
<div class="how-footer-sub">Let's classify waste and make a greener planet together!</div>
</div>
</div>

    """, unsafe_allow_html=True)
 
elif page == "ℹ️ About":
 
    st.title("♻️ About The Project")
 
    st.markdown("""

## Waste Image Classification
 
Waste segregation is a critical step in effective recycling and environmental sustainability.
 
Traditionally, waste sorting:
 
- Requires significant manual labor

- Is time consuming

- Increases operational costs

- Can expose workers to harmful materials

- Is prone to human error
 
This project uses a ResNet50 model to automatically identify waste categories from images.
 
### Categories Supported
 
- Cardboard

- Glass

- Metal

- Paper

- Plastic

- Trash
 
### Project Goal
 
To make waste segregation faster, safer, and more accurate using Artificial Intelligence.
 
### Benefits
 
✅ Reduced manpower requirements
 
✅ Lower operational costs
 
✅ Faster sorting process
 
✅ Improved recycling efficiency
 
✅ Safer working conditions
 
✅ Cleaner environment
 
This system can support smart recycling initiatives and contribute towards sustainable waste management.

""")
 