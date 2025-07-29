import streamlit as st
import cv2
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
import os
import re
import json
import time
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# --- Setup ---
st.set_page_config(page_title="Real-time License Plate Detection (WebRTC)", layout="wide")

@st.cache_resource
def load_models():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    processor = TrOCRProcessor.from_pretrained('ziyadazz/OCR-PLAT-NOMOR-INDONESIA')
    ocr_model = VisionEncoderDecoderModel.from_pretrained('ziyadazz/OCR-PLAT-NOMOR-INDONESIA').to(device)
    yolo_model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt', device=device)
    return yolo_model, processor, ocr_model, device

model, processor, ocr_model, device = load_models()

with open('example.json', 'r') as f:
    plate_data = json.load(f)
    known_plates = [item['plat'] for item in plate_data]

try:
    font = ImageFont.truetype("arial.ttf", 40)
except IOError:
    font = ImageFont.load_default()

def check_plate_match(ocr_text, known_plates):
    max_match_len = 0
    for plate in known_plates:
        for i in range(len(ocr_text)):
            for j in range(i, len(ocr_text)):
                substring = ocr_text[i:j+1]
                if substring in plate and len(substring) > max_match_len:
                    max_match_len = len(substring)
    return max_match_len

class VideoTransformer(VideoTransformerBase):
    def __init__(self):
        self.last_process_time = 0

    def transform(self, frame):
        current_time = time.time()
        if current_time - self.last_process_time < 0.5:
            return frame.to_ndarray(format="bgr24")

        self.last_process_time = current_time
        
        image = Image.fromarray(frame.to_ndarray(format="rgb24"))
        draw = ImageDraw.Draw(image)
        results = model(image)

        for *box, conf, cls in results.xyxy[0]:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            plate_img = image.crop((x1, y1, x2, y2))
            
            pixel_values = processor(images=plate_img, return_tensors="pt").pixel_values.to(device)
            generated_ids = ocr_model.generate(pixel_values)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            if generated_text:
                clean_text = re.sub(r'[^A-Z0-9]', '', generated_text.upper())
                if clean_text:
                    match_len = check_plate_match(clean_text, known_plates)
                    box_color = "red" if match_len >= 5 else "yellow" if match_len == 4 else "green"
                    
                    draw.rectangle([x1, y1, x2, y2], outline=box_color, width=3)
                    display_text = f"{clean_text} ({conf:.2f})"
                    text_position = (x1, y1 - 45)
                    draw.text(text_position, display_text, fill="white", font=font, stroke_fill="black", stroke_width=2)

        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

st.title("Real-time License Plate Detection (WebRTC)")
st.write("This application uses your webcam to detect and read license plates in real-time.")

webrtc_streamer(key="example", video_transformer_factory=VideoTransformer)
