import os

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src import config
from src.video_processor import PoseProcessor

st.set_page_config(page_title="Rehab Pose Correction", layout="centered")
st.title("Pose mapper check")

available = [n for n, p in config.MLP_WEIGHTS.items() if os.path.exists(p)]
if not available:
    st.error(f"No weights found in {config.MODELS_DIR}. Expected: {[p.name for p in config.MLP_WEIGHTS.values()]}")
    st.stop()
model_name = st.radio("Mapper model:", available, horizontal=True)

ctx = webrtc_streamer(
    key="pose",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=PoseProcessor,
    async_processing=True,
)

if ctx.video_processor:
    ctx.video_processor.model_name = model_name
    if ctx.video_processor.error:
        st.error(ctx.video_processor.error)