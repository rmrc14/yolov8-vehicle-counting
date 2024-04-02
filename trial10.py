# -*- coding: utf-8 -*-
"""trial10.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1gRqbLa28KNY1eeB94XS8Z1mpVX9SwbP7
"""



# Commented out IPython magic to ensure Python compatibility.
# Object Detecion
import cv2
from ultralytics import YOLO
#plots
import matplotlib.pyplot as plt
import seaborn as sns

#basics
import pandas as pd
import numpy as np
import os
import subprocess

from tqdm.notebook import tqdm

# Display image and videos
import IPython
from IPython.display import Video, display
from IPython.display import display, Image
# %matplotlib inline

#from google.colab import drive
#drive.mount('/content/drive')

#import gdown
#import cv2
#import os

# Download the video
#video_link = 'https://drive.google.com/uc?export=download&id=19Y4pXFIxgJHeyMnGw0Dd_38OCj6CGEPo'
#output_file = '/content/video1.mp4'
#gdown.download(video_link, output_file, quiet=False)

# Load the video from the current directory
#video_path = "/content/video1.mp4"  # Path to the downloaded video
video_capture = cv2.VideoCapture(vehicle-counting.mp4)

# Check if the video was successfully loaded
if not video_capture.isOpened():
    raise ValueError(f"Error opening video file: {video_path}")

# Get video properties
fps = video_capture.get(cv2.CAP_PROP_FPS)
frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
total_frames = int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
print("FPS:", fps)
print("HEIGHT:", frame_height)
print("WIDTH:", frame_width)
print("total frames:", total_frames)

# Set the frame position to the desired frame (e.g., frame number 100)
frame_number = 100
video_capture.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

# Read the frame
ret, frame = video_capture.read()

# Check if the frame was successfully read
if not ret:
    raise ValueError(f"Error reading frame {frame_number} from the video.")
# Convert the frame from BGR to RGB (OpenCV uses BGR by default)
frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

# Display the frame
display(Image(data=cv2.imencode('.jpg', frame_rgb)[1].tobytes()))

# Release the video capture
video_capture.release()

# Auxiliary functions
def risize_frame(frame, scale_percent):
    """Function to resize an image in a percent scale"""
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)

    # resize image
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    return resized

#loading a YOLO model
model = YOLO('yolov8n.pt')

#geting names from classes
dict_classes = model.model.names

### Configurations
# Verbose during prediction
verbose = False
# Scaling percentage of the original frame
scale_percent = 50

# -------------------------------------------------------
# Reading video with cv2
video_capture = cv2.VideoCapture(video_path)

# Objects to detect YOLO
class_IDS = [2, 3,  5, 7]

# Auxiliary variables
centers_old = {}
centers_new = {}
obj_id = 0
vehicle_measure_in = dict.fromkeys(class_IDS, 0)
vehicle_measure_out = dict.fromkeys(class_IDS, 0)
end = []
frames_list = []
cy_line = int(1500 * scale_percent / 100)
cx_direction = int(2000 * scale_percent / 100)
offset = int(8 * scale_percent / 100)
measure_in = 0
measure_out = 0
print(f'[INFO] - Verbose during Prediction: {verbose}')

# Original information of the video
height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
fps = video_capture.get(cv2.CAP_PROP_FPS)
print('[INFO] - Original Dim: ', (width, height))

# Scaling Video for better performance
if scale_percent != 100:
    print('[INFO] - Scaling change may cause errors in pixels lines ')
    width = int(width * scale_percent / 100)
    height = int(height * scale_percent / 100)
    print('[INFO] - Dim Scaled: ', (width, height))

# -------------------------------------------------------
### Video output ####
video_name = 'result.mp4'
output_path = "rep_" + video_name
tmp_output_path = "tmp_" + output_path
VIDEO_CODEC = "MP4V"

output_video = cv2.VideoWriter(tmp_output_path,
                               cv2.VideoWriter_fourcc(*VIDEO_CODEC),
                               fps, (width, height))

# -------------------------------------------------------
# Executing Recognition
for i in tqdm(range(int(video_capture.get(cv2.CAP_PROP_FRAME_COUNT)))):

    # Reading frame from video
    _, frame = video_capture.read()

    # Applying resizing of read frame
    frame = risize_frame(frame, scale_percent)

    if verbose:
        print('Dimension Scaled(frame): ', (frame.shape[1], frame.shape[0]))

    # Getting predictions
    y_hat = model.predict(frame, conf=0.7, classes=class_IDS, device=0, verbose=False)

    # Getting the bounding boxes, confidence, and classes of the recognized objects in the current frame.
    boxes = y_hat[0].boxes.xyxy.cpu().numpy()
    conf = y_hat[0].boxes.conf.cpu().numpy()
    classes = y_hat[0].boxes.cls.cpu().numpy()

    # Storing the above information in a dataframe
    positions_frame = pd.DataFrame(y_hat[0].cpu().numpy().boxes.data,
                                   columns=['xmin', 'ymin', 'xmax', 'ymax', 'conf', 'class'])

    # Translating the numeric class labels to text
    labels = [dict_classes[i] for i in classes]

    # Drawing transition line for in\out vehicles counting
    cv2.line(frame, (0, cy_line), (int(4500 * scale_percent / 100), cy_line), (255, 255, 0), 8)

    # For each vehicle, draw the bounding-box and counting each one the pass through the transition line (in\out)
    for ix, row in enumerate(positions_frame.iterrows()):
        # Getting the coordinates of each vehicle (row)
        xmin, ymin, xmax, ymax, confidence, category = row[1].astype('int')

        # Calculating the center of the bounding-box
        center_x, center_y = int(((xmax + xmin)) / 2), int((ymax + ymin) / 2)

        # Drawing center and bounding-box of the vehicle in the given frame
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (255, 0, 0), 5)  # box
        cv2.circle(frame, (center_x, center_y), 5, (255, 0, 0), -1)  # center of box

        # Drawing above the bounding-box the name of class recognized.
        cv2.putText(img=frame, text=labels[ix] + ' - ' + str(np.round(conf[ix], 2)),
                    org=(xmin, ymin - 10), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 0, 0),
                    thickness=2)

        # Checking if the center of recognized vehicle is in the area given by the transition line + offset and transition line - offset
        if (center_y < (cy_line + offset)) and (center_y > (cy_line - offset)):
            if (center_x >= 0) and (center_x <= cx_direction):
                measure_in += 1
                vehicle_measure_in[category] += 1
            else:
                measure_out += 1
                vehicle_measure_out[category] += 1

    # Updating the counting type of vehicles
    measure_in_plt = [f'{dict_classes[k]}: {i}' for k, i in vehicle_measure_in.items()]
    measure_out_plt = [f'{dict_classes[k]}: {i}' for k, i in vehicle_measure_out.items()]

    # Drawing the number of vehicles in\out
    cv2.putText(img=frame, text='N. vehicles In',
                org=(30, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                fontScale=1, color=(255, 255, 0), thickness=1)

    cv2.putText(img=frame, text='N. vehicles Out',
                org=(int(2800 * scale_percent / 100), 30),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0), thickness=1)

    # Drawing the counting of type of vehicles in the corners of frame
    xt = 40
    for txt in range(len(measure_in_plt)):
        xt += 30
        cv2.putText(img=frame, text=measure_in_plt[txt],
                    org=(30, xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 255, 0), thickness=1)

        cv2.putText(img=frame, text=measure_out_plt[txt],
                    org=(int(2800 * scale_percent / 100), xt), fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                    fontScale=1, color=(255, 255, 0), thickness=1)

    # Drawing the number of vehicles in\out
    cv2.putText(img=frame, text=f'In:{measure_in}',
                org=(int(1820 * scale_percent / 100), cy_line + 60),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0), thickness=2)

    cv2.putText(img=frame, text=f'Out:{measure_out}',
                org=(int(1800 * scale_percent / 100), cy_line - 40),
                fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(255, 255, 0), thickness=2)

    if verbose:
        print(measure_in, measure_out)
    # Saving frames in a list
    frames_list.append(frame)
    # Saving transformed frames in an output video format
    output_video.write(frame)

# Releasing the video
output_video.release()

#### Post-processing
# Fixing video output codec to run in the notebook\browser
if os.path.exists(output_path):
    os.remove(output_path)

subprocess.run(
    ["ffmpeg", "-i", tmp_output_path, "-crf", "18", "-preset", "veryfast", "-hide_banner", "-loglevel", "error",
     "-vcodec", "libx264", output_path])
os.remove(tmp_output_path)

#output video result
frac = 0.7
Video(data='rep_result.mp4', embed=True, height=int(720 * frac), width=int(1280 * frac))