import streamlit as st
from deep_sort_realtime.deepsort_tracker import DeepSort
import config
from utils1 import load_model, infer_uploaded_video
from pathlib import Path
# Setting page layout
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

# initialize DeepSort object
deepsort = DeepSort()

# image/video options
st.sidebar.header("Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

source_img = None

if source_selectbox == config.SOURCES_LIST[1]: # Video

    infer_uploaded_video(confidence, model, deepsort)

else:
    st.error("Currently only  'Video' source are implemented")

# update code to count vehicle
def update_vehicle_count(bboxes, labels):
    """
    Update the vehicle count based on the bounding boxes and labels.
    :param bboxes: A list of bounding boxes.
    :param labels: A list of labels.
    :return: None
    """
 
    global centers_old
    global centers_new
    for bbox, label in zip(bboxes, labels):
        if label == "car":
            # Get the center of the bounding box
            center = (bbox.x1 + bbox.x2) // 2, (bbox.y1 + bbox.y2) // 2

            # If the bounding box is new, add it to the list of tracked vehicles
            if bbox.track_id not in centers_old:
                centers_old[bbox.track_id] = center
                centers_new[bbox.track_id] = center
            else:
                # If the bounding box is already tracked, update its center
                centers_new[bbox.track_id] = center

            # Check if the vehicle has crossed the lane boundary
            if center[0] < 0:
                # Vehicle has crossed the left lane boundary
                if bbox.track_id in centers_old:
                    del centers_old[bbox.track_id]
                    OBJECT_COUNTER += 1
            else:
                # Vehicle has crossed the right lane boundary
                if bbox.track_id in centers_old:
                    del centers_old[bbox.track_id]
                    config.OBJECT_COUNTER1 += 1


# display vehicle count
st.write("Vehicle In:", config.OBJECT_COUNTER1)
st.write("Vehicle Out:",config.OBJECT_COUNTER)
