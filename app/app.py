import streamlit as st
from PIL import Image
from utils.utils import generate_caption, segment_image

st.set_page_config(
    page_title="VisioAI: Image Magic",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Rubik:wght@500;700&display=swap');
    
    html, body, .stApp {
      background: linear-gradient(110deg, #e3f2fd, #f8fafc 70%);
      font-family: 'Rubik', sans-serif;
    }

    .main-title {
      font-size: 3.2rem !important;
      font-weight: 700 !important;
      color: #1259be;
      text-shadow: 1px 1px 0 #c3e0fa, 2px 2px 10px #90caf9;
      text-align: center;
      letter-spacing: 0.9px;
      margin-top: 16px;
      margin-bottom: 20px;
    }

    .section-title {
      font-size: 1.35rem !important;
      font-weight: 600 !important;
      color: #1565c0;
      margin-bottom: 18px;
    }
    .caption-box {
      background: rgba(30, 136, 229, .08);
      color: #14539a;
      border-left: 6px solid #1976d2;
      border-radius: 6px;
      font-size: 1.1rem;
      padding: 1rem 1.2rem;
      margin-top: 10px;
      margin-bottom: 5px;
      animation: fadeInScale 0.7s cubic-bezier(.23,.89,.23,.2);
    }
    .seg-img {
      border-radius: 10px;
      border: 3px solid #1976d2;
      box-shadow: 0 3px 24px rgba(25,118,210, 0.08);
      margin-bottom: 12px;
      animation: fadeInScale 0.7s cubic-bezier(.23,.89,.23,.2);
    }
    @keyframes fadeInScale {
       0% { opacity:0; transform:scale(.96);}
       60% { opacity:.93; }
       100% { opacity:1; transform:scale(1);}
    }
    .footer {
      font-size: 1.03rem;
      color: #628bc8;
      text-align:center;
      margin-top: 2.2rem;
      padding: .7em 0 .3em 0;
      letter-spacing: .6px;
    }
    .footer a {
      color: #1976d2; text-decoration:none; font-weight: 500;
    }
    /* Hide default Streamlit header */
    header {visibility: hidden;}
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image('logo.png', width=190)
    st.markdown(
        "<span style='font-size:2rem; "
        "font-weight:700; color:#1259be;'>VisioAI</span><br>"
        "<span style='color:#1976d2;font-size:1.01rem;'>Image Captioning & Segmentation Lab</span>",
        unsafe_allow_html=True
    )
    st.write(" ")
    st.markdown("**How it works:**\n"
        "- Upload a photo\n"
        "- Get a smart caption (AI vision)\n"
        "- See smart segmentation overlay\n"
        "- Download results"
    )
    st.write("© 2025 VisioAI Lab")
    st.write("---")
    st.info("Tip: Use natural everyday or complex images for interesting captions & masks!", icon="💡")


st.markdown("<div class='main-title'>✨ Image Captioning & Segmentation</div>", unsafe_allow_html=True)

st.markdown("<div class='upload-card'>", unsafe_allow_html=True)
st.markdown("<div class='section-title'>Upload an Image</div>", unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    label="Click or drag image file here (jpg, png, jpeg)",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)


if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1.5], gap="large")
    with col1:
        st.markdown("<div class='output-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Uploaded Image</div>", unsafe_allow_html=True)
        st.image(image, caption="Original", use_container_width="always", channels="RGB")
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown("<div class='output-card'>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Generated Caption</div>", unsafe_allow_html=True)
        with st.spinner("AI is reading your image ..."):
            caption = generate_caption(image.copy())
        st.markdown(
            f"<div class='caption-box'>{caption}</div>", 
            unsafe_allow_html=True
        )
        st.markdown("<div class='section-title'>Image Segmentation</div>", unsafe_allow_html=True)
        with st.spinner("Segmenting image ..."):
            segmented = segment_image(image.copy())
        st.image(segmented, caption="Segmentation Output", use_container_width=True, output_format='PNG', channels="RGB")
        st.markdown("</div>", unsafe_allow_html=True)

    with st.container():
        st.markdown("#### ⬇Download Results")
        col_a, col_b = st.columns(2)
        with col_a:
            st.download_button(
                label="Download Caption (.txt)",
                data=caption,
                file_name="caption.txt",
                mime="text/plain"
            )
        with col_b:
            from io import BytesIO
            buf = BytesIO()
            segmented.save(buf, format="PNG")
            st.download_button(
                label="Download Segmentation (.png)",
                data=buf.getvalue(),
                file_name="segmented.png",
                mime="image/png"
            )

else:
    st.markdown("""
        <div style='text-align:center; font-size:1.23rem;'>Please upload an image to begin.<br>
        Supported formats: JPG, JPEG, PNG.<br>
        Try a complex, colorful image and see the magic! </div>
    """, unsafe_allow_html=True)

st.markdown("""
    <div class='footer'>
        <span>
            Built with Accuracy | 
            <a href='mailto:lollyprabhu2004@gmail.com'>Contact</a> | 
            <a href='https://github.com/Rishikesh4089'>GitHub</a>
        </span>
    </div>
""", unsafe_allow_html=True)
