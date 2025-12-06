import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import warnings
from dotenv import load_dotenv
import os

# --- Configuration ---
warnings.filterwarnings('ignore') # Ignore warning messages to keep the output clean
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1' # Enable MPS fallback for Mac users (M1/M2/M3 chips)
load_dotenv() # Load environment variables (like API keys if needed)

# --- Page Setup ---
st.set_page_config(
    page_title="AI Vision Workshop",
    page_icon="💻",
    layout="wide"
)

# Display the main title and subtitle
st.title("AI Workshop")
st.markdown("### Computer vision with AI")

# Sidebar title for controls
st.sidebar.title("Controls")

# --- Model Loading ---
# Load the YOLO object detection model
# @st.cache_resource keeps the model in memory so it doesn't reload every time
@st.cache_resource
def load_model():
    return YOLO("yolo11n.pt")

# Function to pick the best available processor (Apple Silicon, NVIDIA GPU, or CPU)
@st.cache_resource
def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    return "cpu"

# Initialize model and move it to the correct device
try:
    device = get_device()
    model = load_model()
    model.to(device)
    st.sidebar.success("YOLO Model Loaded!")
except Exception as e:
    st.error(f"Error loading model: {e}")


# --- Camera Controls ---
# Create buttons to start and stop the camera
start_button = st.button("Start Camera", type="primary")
stop_button = st.button("Stop Camera")

# Use session state to remember if the camera is running or not
if 'camera_running' not in st.session_state:
    st.session_state.camera_running = False

# Update state based on button clicks
if start_button:
    st.session_state.camera_running = True
if stop_button:
    st.session_state.camera_running = False
    
    
# Create a valid empty space to hold the video feed
frame_placeholder = st.empty()

# --- Main Logic ---
if st.session_state.camera_running:
    # 0 usually refers to the default webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.error("Could not open camera.")
    else:
        # Loop to read frames while the camera is active
        while st.session_state.camera_running:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to read frame.")
                break
            
            # Run the AI model on the current frame
            # verbose=False hides extra print outputs
            results = model(frame, verbose=False)
            
            # Draw boxes around detected objects
            annotated_frame = results[0].plot()
            
            # Convert color from BGR (OpenCV default) to RGB (Streamlit expects)
            annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
            
            # Update the placeholder with the new image
            frame_placeholder.image(annotated_frame, channels="RGB")
            
            # Small pause to not overload the CPU
            time.sleep(0.01)
        
        # Release the camera when done
        cap.release()