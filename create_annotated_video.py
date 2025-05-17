import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import json
import os

def create_annotated_video(frames_dir, output_path, keypoints_json, slow_motion_factor=2):
    try:
        # Check if keypoints file exists
        if not os.path.exists(keypoints_json):
            raise FileNotFoundError(f"Keypoints file not found: {keypoints_json}")
            
        # Load the keypoints data
        with open(keypoints_json, 'r') as f:
            data = json.load(f)
            
        if not data:
            raise ValueError("No data found in keypoints file")
            
        print(f"Loaded {len(data)} frames from keypoints file")
        
        # Check if frames directory exists
        frames_dir = Path(frames_dir)
        if not frames_dir.exists():
            raise FileNotFoundError(f"Frames directory not found: {frames_dir}")
            
        # Initialize YOLOv8 model
        model = YOLO("yolov8n-pose.pt")
        
        # Get the first frame to determine video properties
        first_frame_path = frames_dir / Path(data[0]['image']).name
        print(f"Looking for first frame at: {first_frame_path}")
        
        if not first_frame_path.exists():
            raise FileNotFoundError(f"First frame not found: {first_frame_path}")
            
        first_frame = cv2.imread(str(first_frame_path))
        if first_frame is None:
            raise ValueError(f"Could not read first frame: {first_frame_path}")
            
        height, width = first_frame.shape[:2]
        print(f"Video dimensions: {width}x{height}")
        
        # Initialize video writer with slower frame rate
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        original_fps = 30.0
        slow_motion_fps = original_fps / slow_motion_factor
        print(f"Creating slow motion video at {slow_motion_fps} FPS (original: {original_fps} FPS)")
        
        out = cv2.VideoWriter(
            str(output_path),
            fourcc,
            slow_motion_fps,
            (width, height)
        )
        
        if not out.isOpened():
            raise ValueError(f"Could not create video writer for {output_path}")
        
        print("Creating annotated video...")
        processed_frames = 0
        
        for entry in data:
            frame_id = entry["frame"]
            image_path = frames_dir / Path(entry["image"]).name
            
            # Load frame
            frame = cv2.imread(str(image_path))
            if frame is None:
                print(f"⚠️ Could not load image {image_path}")
                continue
            
            # Get pose keypoints
            results = model.predict(source=frame, show=False, verbose=False)
            
            # Draw keypoints and skeleton
            annotated_frame = results[0].plot()
            
            # Add frame number and slow motion indicator
            cv2.putText(
                annotated_frame,
                f"Frame: {frame_id} (Slow Motion {slow_motion_factor}x)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )
            
            # Write frame to video
            out.write(annotated_frame)
            processed_frames += 1
            
            if frame_id % 10 == 0:
                print(f"Processed frame {frame_id}")
        
        # Release video writer
        out.release()
        
        if processed_frames == 0:
            raise ValueError("No frames were processed successfully")
            
        print(f"✅ Video saved to: {output_path}")
        print(f"Processed {processed_frames} frames successfully")
        print(f"Video duration: {processed_frames/slow_motion_fps:.2f} seconds")
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        raise

if __name__ == "__main__":
    frames_dir = "../frames"  # Directory containing the original frames
    output_path = "annotated_video_slow_motion.mp4"
    keypoints_json = "keypoints.json"
    slow_motion_factor = 2  # Adjust this value to change the slow motion speed (higher = slower)
    
    # Print current working directory and check paths
    print(f"Current working directory: {os.getcwd()}")
    print(f"Frames directory: {os.path.abspath(frames_dir)}")
    print(f"Output path: {os.path.abspath(output_path)}")
    print(f"Keypoints file: {os.path.abspath(keypoints_json)}")
    
    create_annotated_video(frames_dir, output_path, keypoints_json, slow_motion_factor) 