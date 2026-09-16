import streamlit as st
import cv2
import numpy as np
import tempfile
import os
from PIL import Image
from detect_alone import StandaloneDrowsinessDetector

# Optional: live webcam streaming
try:
    from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode  # type: ignore[import]
    STREAMLIT_WEBRTC_AVAILABLE = True
except Exception:
    STREAMLIT_WEBRTC_AVAILABLE = False

st.set_page_config(page_title="Drowsiness Detection", layout="wide")
st.title("🚗 Drowsiness Detection — Streamlit UI")

st.sidebar.header("Configuration")
MODE_OPTIONS = ["Image", "Camera (single shot)", "Video", "RTSP Stream"]
if STREAMLIT_WEBRTC_AVAILABLE:
    MODE_OPTIONS.insert(1, "Live Camera (webcam)")
MODE = st.sidebar.selectbox("Mode", MODE_OPTIONS)

EAR_THRESH = st.sidebar.slider("EAR Threshold", 0.1, 0.6, 0.3, 0.01)
MAR_THRESH = st.sidebar.slider("MAR Threshold", 0.3, 1.0, 0.6, 0.01)
CONSEC_FRAMES = st.sidebar.slider("Consecutive frames (drowsiness)", 1, 60, 15)
YAWN_FRAMES = st.sidebar.slider("Consecutive frames (yawn)", 1, 60, 10)

st.sidebar.markdown("---")
st.sidebar.markdown("Built with MediaPipe + OpenCV. Audio alerts are disabled in headless environments.")

@st.cache_resource
def create_detector():
    return StandaloneDrowsinessDetector()

if 'detector' not in st.session_state:
    st.session_state.detector = create_detector()

detector: StandaloneDrowsinessDetector = st.session_state.detector
# Apply UI config
detector.EAR_THRESH = EAR_THRESH
detector.MAR_THRESH = MAR_THRESH
detector.CONSEC_FRAMES = CONSEC_FRAMES
detector.YAWN_FRAMES = YAWN_FRAMES


# --- Live transformer for streamlit-webrtc ---
if STREAMLIT_WEBRTC_AVAILABLE:
    class DrowsinessTransformer(VideoTransformerBase):
        def __init__(self):
            # Each transformer keeps its own detector so per-stream smoothing works
            self.detector = StandaloneDrowsinessDetector()
            # apply current UI config as initial values
            self.detector.EAR_THRESH = EAR_THRESH
            self.detector.MAR_THRESH = MAR_THRESH
            self.detector.CONSEC_FRAMES = CONSEC_FRAMES
            self.detector.YAWN_FRAMES = YAWN_FRAMES

        def transform(self, frame):
            img = frame.to_ndarray(format="bgr24")
            processed_frame, ear, mar = self.detector.process_frame(img)
            # convert to RGB for browser
            return cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)


def process_image_file(uploaded_file):
    image = Image.open(uploaded_file).convert('RGB')
    frame = np.array(image)[:, :, ::-1]  # RGB -> BGR

    detector.session_data = []
    detector.session_start = __import__('time').time()
    processed_frame, ear, mar = detector.process_frame(frame)

    processed_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
    st.image(processed_rgb, caption=f"EAR: {ear:.3f} | MAR: {mar:.3f}")

    out_pil = Image.fromarray(processed_rgb)
    buf = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
    out_pil.save(buf.name)
    with open(buf.name, 'rb') as f:
        st.download_button("Download processed image", f, file_name='processed.png')

    summary = detector.save_session_data()
    st.success(f"Session saved → {summary}")


def process_camera_single_shot():
    img_file_buffer = st.camera_input("Take a picture")
    if img_file_buffer is not None:
        process_image_file(img_file_buffer)


def process_video_file(uploaded_file, max_frames=None, stride=1):
    tmp_in = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1])
    tmp_in.write(uploaded_file.getbuffer())
    tmp_in.flush()

    cap = cv2.VideoCapture(tmp_in.name)
    if not cap.isOpened():
        st.error("Cannot open video file")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    p = st.progress(0)
    i = 0
    processed = 0

    detector.session_data = []
    detector.session_start = __import__('time').time()

    with st.spinner('Processing video...'):
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            if i % stride != 0:
                continue
            processed_frame, ear, mar = detector.process_frame(frame)
            out.write(processed_frame)
            processed += 1
            if max_frames and processed >= max_frames:
                break
            if total:
                p.progress(min(i / total, 1.0))

    cap.release()
    out.release()

    st.success("Processing complete")
    st.video(tmp_out.name)
    with open(tmp_out.name, 'rb') as f:
        st.download_button("Download processed video", f, file_name='processed.mp4')

    summary = detector.save_session_data()
    st.success(f"Session saved → {summary}")


def process_rtsp_stream(url, duration=10, stride=1):
    cap = cv2.VideoCapture(url)
    if not cap.isOpened():
        st.error("Cannot open stream. Ensure URL is accessible from this machine.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    tmp_out = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    out = cv2.VideoWriter(tmp_out.name, fourcc, fps, (width, height))

    frames_to_capture = int(fps * duration)
    i = 0
    p = st.progress(0)

    detector.session_data = []
    detector.session_start = __import__('time').time()

    with st.spinner('Capturing stream...'):
        while i < frames_to_capture:
            ret, frame = cap.read()
            if not ret:
                break
            i += 1
            if i % stride != 0:
                continue
            processed_frame, ear, mar = detector.process_frame(frame)
            out.write(processed_frame)
            p.progress(min(i / frames_to_capture, 1.0))

    cap.release()
    out.release()

    st.success("Stream capture complete")
    st.video(tmp_out.name)
    with open(tmp_out.name, 'rb') as f:
        st.download_button("Download processed stream", f, file_name='processed_stream.mp4')

    summary = detector.save_session_data()
    st.success(f"Session saved → {summary}")


# Main UI
if MODE == "Image":
    uploaded_file = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        process_image_file(uploaded_file)

elif MODE == "Live Camera (webcam)":
    if not STREAMLIT_WEBRTC_AVAILABLE:
        st.error("streamlit-webrtc is not installed. Install it with `pip install streamlit-webrtc` and restart the app.")
    else:
        st.write("Live webcam — processing frames in real time. Press 'Stop' to end the stream.")
        webrtc_streamer(key="live", mode=WebRtcMode.SENDRECV, video_transformer_factory=DrowsinessTransformer)

elif MODE == "Camera (single shot)":
    process_camera_single_shot()

elif MODE == "Video":
    uploaded_file = st.file_uploader("Upload a video file", type=['mp4', 'mov', 'avi', 'mkv'])
    col1, col2 = st.columns(2)
    with col1:
        max_frames = st.number_input('Max frames to process (0 = all)', min_value=0, value=0, step=1)
        stride = st.number_input('Process every N-th frame (stride)', min_value=1, value=1, step=1)
    with col2:
        if uploaded_file is not None:
            if st.button('Start processing'):
                process_video_file(uploaded_file, max_frames if max_frames > 0 else None, stride)

elif MODE == "RTSP Stream":
    url = st.text_input('RTSP / Stream URL')
    duration = st.slider('Capture duration (s)', 5, 60, 10)
    stride = st.number_input('Process every N-th frame (stride)', min_value=1, value=1, step=1)
    if st.button('Capture stream'):
        if url:
            process_rtsp_stream(url, duration, stride)
        else:
            st.error('Please enter a stream URL')


st.markdown('---')
st.markdown('### Notes')
st.markdown('- Use **Live Camera (webcam)** for real-time browser camera streaming (requires `streamlit-webrtc`).')
st.markdown('- Use **Camera (single shot)** for quick checks via your browser camera.')
st.markdown('- Run locally with `streamlit run streamlit_app.py`.')
