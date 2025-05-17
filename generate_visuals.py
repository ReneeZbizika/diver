import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
from pathlib import Path
import torch
from ultralytics import YOLO
import sys
import os

def calculate_com(keypoints):
    """Calculate center of mass from keypoints."""
    # YOLOv8 keypoints are in format [x, y, confidence]
    valid_points = [kp for kp in keypoints if kp[2] > 0.5]  # Filter by confidence
    if not valid_points:
        return None
    return np.mean(valid_points, axis=0)[:2]  # Only use x,y coordinates

def process_video(video_path, output_dir):
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Check if video exists
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        # Initialize YOLOv8 model
        model = YOLO("yolov8n-pose.pt")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(
            str(output_dir / 'annotated_video.mp4'),
            fourcc,
            fps,
            (width, height)
        )
        
        # Initialize lists for COM trajectory
        com_trajectory = []
        frame_data = []
        frame_count = 0
        
        print("Processing video...")
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Get pose keypoints
            results = model.predict(source=frame, show=False, verbose=False)
            keypoints = results[0].keypoints.data[0].cpu().numpy()  # Get keypoints for first person
            
            # Save keypoints to JSON
            frame_data.append({
                'frame': frame_count,
                'keypoints': keypoints.tolist()
            })
            
            # Calculate COM
            com = calculate_com(keypoints)
            if com is not None:
                com_trajectory.append(com)
            
            # Draw keypoints on frame
            annotated_frame = results[0].plot()
            out.write(annotated_frame)
            
            frame_count += 1
            if frame_count % 10 == 0:
                print(f"Processed {frame_count} frames...")
        
        # Release video resources
        cap.release()
        out.release()
        
        # Save keypoints to JSON
        print("Saving keypoints to JSON...")
        with open(output_dir / 'keypoints.json', 'w') as f:
            json.dump(frame_data, f, indent=2)
        
        # Plot COM trajectory
        if com_trajectory:
            print("Generating COM trajectory plot...")
            com_trajectory = np.array(com_trajectory)
            plt.figure(figsize=(10, 6))
            plt.plot(com_trajectory[:, 0], com_trajectory[:, 1], 'b-', label='COM Trajectory')
            plt.title('Center of Mass Trajectory')
            plt.xlabel('X Position')
            plt.ylabel('Y Position')
            plt.legend()
            plt.grid(True)
            plt.savefig(output_dir / 'com_trajectory.png')
            plt.close()
            
        print("Processing complete! Outputs saved in:", output_dir)
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    video_path = 'videos/gymnast.mp4'  # Update with your video path
    output_dir = 'output'
    process_video(video_path, output_dir) 