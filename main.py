import cv2
import shutil
from paddleocr import PaddleOCR
import logging
import os
from cnocr import CnOcr

# Initialize PaddleOCR with GPU
ocr = PaddleOCR(use_gpu=True)
logging.disable(logging.CRITICAL)

def ocr_output(image_name) :
    ocr = PaddleOCR(use_gpu=True) 
    img_path = image_name # replace 'test.jpg' with your image
    result = ocr.ocr(img_path)
    for line in result:
        for word_info in line:
            return word_info[-1]


# Open the video file
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

# Get frames per second and frame count
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_directory = 'frames'

# Delete the output directory if it already exists
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
os.makedirs(output_directory, exist_ok=True)





# Loop through frames and perform OCR
for frame_index in range(frame_count):
    # Capture the frame
    ret, frame = cap.read()
    if not ret:
        break

    frame_filename = f"frames/frame_{frame_index + 1}.jpg"
    cv2.imwrite(frame_filename, frame)

    # # Perform OCR on the frame
    result = ocr_output(frame)

    print("Frame"+ str(frame_index), result)

# Release the video capture
cap.release()
