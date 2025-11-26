#hepl code i used to extract <= 100 representative frames from video file and save them as images

from pathlib import Path
import cv2
import os 
import argparse

def get_args():
    parser = argparse.ArgumentParser(description='extract frames from video') 
    parser.add_argument('--input', type=str, default=None, help='path of the input video file')
    parser.add_argument('--save_directory', type=str, default=None, help='path where to save the results')
    parser.add_argument('--divider', type=int, default=8, help='divider to select frames')
    return parser.parse_args()

args = get_args()
input = args.input
save_directory = args.save_directory
frame_count_divider = args.divider # Save every 8th frame to get approximately 100 frames from my video.
                                                        # Will vary based on input video length

cap = cv2.VideoCapture(input)

if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
  
frame_count = 0
# Loop through frames
while cap.isOpened():

    ret, frame = cap.read()  
    
    if not ret:  
        break

    file_name = os.path.splitext(os.path.basename(input))[0]
    save_path = os.path.join(save_directory, f"frame_{frame_count}.jpg")
    frame_count +=1
    if frame_count % frame_count_divider == 0:
        cv2.imwrite(str(save_path), frame)