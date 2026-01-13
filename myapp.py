import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import tempfile
import time
from model_loader import download_if_missing

st.set_page_config(layout="wide")

@st.cache_resource
def load_models():
    print('inside model loader')
    baseline_path = download_if_missing(
        "baseline.pt",
        "https://github.com/moom-03/App-for-object-detection-of-pizza-toppings-/releases/download/v1.0-models/baseline.pt"
    )

    improved_path = download_if_missing(
        "improved.pt",
        "https://github.com/moom-03/App-for-object-detection-of-pizza-toppings-/releases/download/v1.0-models/improved.pt"
    )
    return YOLO(baseline_path), YOLO(improved_path)

model_base, model_best = load_models()
print('models loaded')
st.title("YOLOv8 Model Comparison")

conf = st.slider("Confidence threshold", 0.01, 1.0, 0.11)


input_type = st.radio("Select input type", ["Image", "Video"])

# =========================
# Session state init
# =========================
if "paused" not in st.session_state:
    st.session_state.paused = True

if "frame_idx" not in st.session_state:
    st.session_state.frame_idx = 0

if "video_path" not in st.session_state:
    st.session_state.video_path = None

if "last_frame_base" not in st.session_state:
    st.session_state.last_frame_base = None

if "last_frame_best" not in st.session_state:
    st.session_state.last_frame_best = None

# ===============
# IMAGE PIPELINE
# ===============
if input_type == "Image":
    uploaded = st.file_uploader("Upload image", type=["jpg", "jpeg", "png"])
    if uploaded is not None:
        # Decode image
        img_bytes = np.frombuffer(uploaded.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        # Inference
        res_base = model_base(img, conf=conf)[0]
        res_best = model_best(img, conf=conf)[0]

        # Display
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baseline")
            st.image(res_base.plot(), channels="BGR")
        with col2:
            st.subheader("Improved")
            st.image(res_best.plot(), channels="BGR")



# ==================
# VIDEO PIPELINE 
# ==================
else:
    uploaded_video = st.file_uploader(
        "Upload video",
        type=["mp4", "mov", "avi", "mkv"],
    )

    # -------------------------
    # Playback controls
    # -------------------------
    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns(3)

    with col_ctrl1:
        if st.button("▶ Play"):
            st.session_state.paused = False

    with col_ctrl2:
        if st.button("⏸ Pause"):
            st.session_state.paused = True

    with col_ctrl3:
        if st.button("New video (rerun)"):
            st.session_state.clear()
            st.rerun()

    # -------------------------
    # Video loading
    # -------------------------
    if uploaded_video is not None:
        if st.session_state.video_path is None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_video.read())
            st.session_state.video_path = tfile.name

        cap = cv2.VideoCapture(st.session_state.video_path)
        cap.set(cv2.CAP_PROP_POS_FRAMES, st.session_state.frame_idx)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Baseline (video)")
            baseline_placeholder = st.empty()

        with col2:
            st.subheader("Improved (video)")
            improved_placeholder = st.empty()

        # -------------------------
        # Frame loop
        # -------------------------
        while cap.isOpened():
            # If paused → just show last frame
            if st.session_state.paused:
                if st.session_state.last_frame_base is not None:
                    baseline_placeholder.image(
                        st.session_state.last_frame_base, channels="BGR"
                    )
                    improved_placeholder.image(
                        st.session_state.last_frame_best, channels="BGR"
                    )
                break

            ret, frame = cap.read()
            if not ret:
                break

            res_base = model_base(frame, conf=conf)[0]
            res_best = model_best(frame, conf=conf)[0]

            frame_base = res_base.plot()
            frame_best = res_best.plot()

            st.session_state.last_frame_base = frame_base
            st.session_state.last_frame_best = frame_best

            baseline_placeholder.image(frame_base, channels="BGR")
            improved_placeholder.image(frame_best, channels="BGR")

            st.session_state.frame_idx += 1

            time.sleep(0.03)

        cap.release()
