from pathlib import Path
import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort
import config
from utils1 import load_model, infer_uploaded_video

# setting page layout
st.set_page_config(
    page_title="vehicle counting using- YOLOv8",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# main page heading
st.title("vehicle counting using- YOLOv8")

# sidebar
st.sidebar.header("DL Model Config")
model_type = None
model_type = st.sidebar.selectbox(
        "Select Model",
        config.DETECTION_MODEL_LIST
    )



confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100

model_path = ""
if model_type:
    model_path = Path(config.DETECTION_MODEL_DIR, str(model_type))
else:
    st.error("Please Select Model in Sidebar")
# load pretrained DL model
try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model. Please check the specified path: {model_path}")

# image/video options
st.sidebar.header("Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None

if source_selectbox == config.SOURCES_LIST[1]: # Video
    infer_uploaded_video(confidence, model)

else:
    st.error("Currently only  'Video' source are implemented")