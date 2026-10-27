import os

import streamlit as st
from streamlit_webrtc import WebRtcMode, webrtc_streamer

from src import config
from src.video_processor import PoseProcessor

st.set_page_config(page_title="Rehab Pose Correction", layout="centered")
st.title("Physiotherapy coach")

available = [n for n, p in config.MLP_WEIGHTS.items() if os.path.exists(p)]
if not available:
    st.error(f"No mapper weights found in {config.MODELS_DIR}. Expected: {[p.name for p in config.MLP_WEIGHTS.values()]}")
    st.stop()

selected_ex = st.selectbox("Exercise:", list(config.EXERCISES), format_func=lambda k: config.EXERCISES[k])

error_placeholder = st.empty()
info_placeholder = st.empty()

ctx = webrtc_streamer(
    key="pose",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=PoseProcessor,
    async_processing=True,
)

if ctx.video_processor:
    proc = ctx.video_processor
    proc.model_name = config.DEFAULT_MODEL
    proc.exercise = selected_ex

    if proc.error:
        error_placeholder.error(proc.error)
    else:
        error_placeholder.empty()

    if proc.coach_info:
        (info_placeholder.info if proc.coach else info_placeholder.error)(proc.coach_info)

    st.caption("Stand in the starting pose and begin. Press Restart to start tracking from the beginning again.")
    if st.button("Restart exercise"):
        proc.request_restart()