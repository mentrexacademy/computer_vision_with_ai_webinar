import streamlit as st
import cv2
import time
import numpy as np
from ultralytics import YOLO
import torch
import warnings
from dotenv import load_dotenv
import os
import mediapipe as mp
import base64
from groq import Groq

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
def load_models():
    model = YOLO("yolo11n.pt")
    mp_hands = mp.solutions.hands # load mediapipe Hand model
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5
    )
    return model, hands

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
    model, hands = load_models()
    model.to(device)
    st.sidebar.success("Models Loaded!")
except Exception as e:
    st.error(f"Error loading models: {e}")

def predict_gender(frame):
    # Load API KEY
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key: return "No API Key"
    
    client = Groq(api_key=api_key)
    _, buffer = cv2.imencode('.jpg', frame)
    img_str = base64.b64encode(buffer).decode('utf-8')
    
    try:
        # Use the LLM Image model to predict Gender
        completion = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Predict gender. Return ONLY 'Male' or 'Female'."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_str}"}}
                ]
            }],
            temperature=0.1,
            max_tokens=10
        )
        return completion.choices[0].message.content
    except:
        return "Error"

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
mp_drawing = mp.solutions.drawing_utils

# --- Main Logic ---
if st.session_state.camera_running:
    # 0 usually refers to the default webcam
    cap = cv2.VideoCapture(0)
    gender_text = "Waiting..."
    frame_count = 0
    
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
            results = model(frame, verbose=False, classes=[0])
            annotated_frame = results[0].plot()
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            hand_results = hands.process(rgb_frame)
            
            # Hand landmark 
            if hand_results.multi_hand_landmarks:
                for landmarks in hand_results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(annotated_frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS)
            
            # use LLM to predict Gender
            if frame_count % 30 == 0:
                if len(results[0].boxes) > 0:
                    gender_text = predict_gender(frame)
            
            # put the predicted text on the screen
            cv2.putText(annotated_frame, f"Gender: {gender_text}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 4, (0, 0, 255), 5)
            
            frame_placeholder.image(annotated_frame, channels="BGR")
            frame_count += 1
            # Small pause to not overload the CPU
            time.sleep(0.01)
        
        # Release the camera when done
        cap.release()