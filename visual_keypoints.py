import json
import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load JSON
with open("keypoints.json", "r") as f:
    data = json.load(f)

output_dir = Path("outputs/annotated_frames")
output_dir.mkdir(parents=True, exist_ok=True)

def draw_keypoints(image, keypoints, threshold=0.5, radius=4, color=(0, 255, 0)):
    for x, y, conf in keypoints:
        if conf > threshold and x > 0 and y > 0:
            cv2.circle(image, (int(x), int(y)), radius, color, -1)
    return image

# First, let's process the keypoints data
keypoints_list = []
valid_frames = []

for entry in data:
    frame_id = entry["frame"]
    image_path = entry["image"]
    keypoints = entry["keypoints"]

    # Skip frames without keypoints or with low confidence
    if not keypoints or all(conf < 0.3 for _, _, conf in keypoints):
        continue

    # Add to our lists
    keypoints_list.append(keypoints)
    valid_frames.append(entry)

    # Try to load and process the image
    try:
        # Get the absolute path to the image
        img_path = Path("../frames") / Path(image_path).name
        img = cv2.imread(str(img_path))
        
        if img is not None:
            # Draw keypoints
            img = draw_keypoints(img, keypoints)
            # Save annotated frame
            out_path = output_dir / f"frame_{frame_id}.jpg"
            cv2.imwrite(str(out_path), img)
            print(f"✅ Saved: {out_path}")
        else:
            print(f"⚠️ Could not load image {img_path}")
    except Exception as e:
        print(f"⚠️ Error processing frame {frame_id}: {str(e)}")

# Now create the trajectory plot
plt.figure(figsize=(15, 8))

# Get the maximum number of keypoints
max_keypoints = max(len(kp) for kp in keypoints_list)

# Plot each keypoint type
for kp_idx in range(max_keypoints):
    x_coords = []
    y_coords = []
    
    # Collect coordinates for this keypoint across all frames
    for frame_kps in keypoints_list:
        if kp_idx < len(frame_kps):
            x, y, conf = frame_kps[kp_idx]
            if conf > 0.3:  # Only use high confidence points
                x_coords.append(x)
                y_coords.append(y)
    
    if x_coords and y_coords:  # Only plot if we have valid points
        plt.plot(x_coords, y_coords, label=f'Keypoint {kp_idx+1}')

plt.title('Keypoint Trajectories')
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.legend()
plt.grid(True)
plt.savefig('keypoint_trajectories.png')
plt.close()

print("✅ Processing complete!")
