import streamlit as st
import pandas as pd
st.header ("vehicle count using YOLO v8")
s1= st.sidebar.selectbox('choose type',
                          ['select','image','video'])
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100




