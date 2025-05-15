"""realtime.py


Use for real time inference


Load model from models/ and use cv2 and yolov8 to process and analyze each frame


For now, we use the example video gymnast.mp4


"""


from ultralytics import YOLO
import cv2
import numpy as np


# Load in model
model = YOLO("yolov8n-pose.pt")  # or your custom trained model


# Path to video as mp4
name = "gymnast"
path_to_video = f'videos/{name}.mp4'


# capture video
capture = cv2.VideoCapture(path_to_video)


# save pose video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')


# initialize frame number to 0
f = 0


ret, frame = capture.read()
if not ret:
   print("Failed to read first frame.")
   exit()
  
frame_height, frame_width = frame.shape[:2]
fps = capture.get(cv2.CAP_PROP_FPS) or 20.0  # fallback to 20 if FPS is unknown
out = cv2.VideoWriter(f"{name}_pose_output.mp4", fourcc, fps, (frame_width, frame_height))


while(capture.isOpened):
   ret, frame = capture.read()
   if ret == False:
       break
  
   #cv2.imwrite(f'captures/example_{f}.jpg', frame) # save video frame
  
   results = model.predict(source=frame, show=False, verbose=False)
   annotated_frame = results[0].plot()
      
   out.write(annotated_frame)
  
   cv2.imshow("Pose Detection", annotated_frame)
  
   f += 1
  
out.release()
capture.release()
cv2.destroyAllWindows()


print("done")

