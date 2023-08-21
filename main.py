import cv2
import shutil
import numpy as np
from paddleocr import PaddleOCR
import logging
import os
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
# from cnocr import CnOcr


font = ImageFont.truetype('/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc',30)
# Initialize PaddleOCR with GPU
ocr = PaddleOCR(use_gpu=True)
logging.disable(logging.CRITICAL)

def ocr_output(image_name) :
    ocr = PaddleOCR(use_gpu=True,lang='ch',det_limit_side_len=4200,det_db_thresh=0.1,det_db_unclip_ratio=3.0) 
    img_path = image_name # replace 'test.jpg' with your image
    result = ocr.ocr(img_path)
    words = []
    # print(result)
    for line in result:
        for word_info in line:
            words.append(word_info[-1][0])
    # print(words)
    return [word_info[0] for line in result for word_info in line],' '.join(words)


def ocr_det(image_name) :
    ocr = PaddleOCR(use_gpu=True,lang='ch',rec=False) 
    img_path = image_name # replace 'test.jpg' with your image
    result = ocr.ocr(img_path)
    # words = []
    # # print(result)
    # for line in result:
    #     for word_info in line:
    #         words.append(word_info[-1][0])
    # print(words)
    return [word_info[0] for line in result for word_info in line]#,' '.join(words)


# Open the video file
video_path = 'test.mp4'
cap = cv2.VideoCapture(video_path)

# Get frames per second and frame count
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

output_directory = 'frames1'

# Delete the output directory if it already exists
if os.path.exists(output_directory):
    shutil.rmtree(output_directory)
os.makedirs(output_directory, exist_ok=True)



xp = [0, 64, 128, 192, 255]
fp = [0, 16, 64, 128, 255]
x = np.arange(256)
table = np.interp(x, xp, fp).astype('uint8')


def highpass(img, sigma):
    return img - cv2.GaussianBlur(img, (0,0), sigma) + 127

# Loop through frames and perform OCR
for frame_index in tqdm([*range(0,frame_count)]):
    # Capture the frame
    ret, frame = cap.read()
    if frame_index < 0:
        continue
    # frame = cv2.LUT(frame, table)
    # frame = highpass(frame,4)
    # sharpen_filter=np.array([[-1,-1,-1],
    #              [-1,9,-1],
    #             [-1,-1,-1]])
    
    # frame=cv2.filter2D(frame,-1,sharpen_filter)
    # frame=cv2.filter2D(frame,-1,sharpen_filter)
    # frame=cv2.filter2D(frame,-1,sharpen_filter)

    if not ret:
        break
    # if frame_index > 3:
    #     break
    
    # print(frame.shape)
    pols,result = ocr_output(frame)
    for pol in pols:
        pola = np.array(pol + [pol[0]]).astype(np.int32)
        # print(pola.shape)
        cv2.polylines(frame, [pola], False, (0,0,255))
        img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
        drw = ImageDraw.Draw(img)
        drw.text((0,0), result,fill='red',font=font)
        frame = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    frame_filename = f"frames/frame_{str(frame_index).zfill(5)}.jpg"
    cv2.imwrite(frame_filename, frame)
    # pols = ocr_det(frame)
    # for pol in pols:
    #     pola = np.array(pol + [pol[0]]).astype(np.int32)
    #     # print(pola.shape)
    #     cv2.polylines(frame, [pola], False, (0,0,255))
    #     # img = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))
    #     # drw = ImageDraw.Draw(img)
    #     # drw.text((0,0), result,fill='red',font=font)
    #     # frame = cv2.cvtColor(np.array(img),cv2.COLOR_RGB2BGR)
    frame_filename = f"frames1/frame_{str(frame_index).zfill(5)}.jpg"
    cv2.imwrite(frame_filename, frame)

    # # Perform OCR on the frame

    print("Frame"+ str(frame_index), pols, result)

# Release the video capture
cap.release()
