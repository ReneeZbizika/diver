import cv2

# path to video as mp4
path_to_vdeo = 'diver/videos/example'

# capture video
capture = cv2.VideoCapture()

# initialize frame number to 0
f = 0

while(capture.isOpened):
    ret, frame = capture.read()
    if ret == False:
        break
    
    # to save video
    cv2.imwrite('new_path_for_one_frame.jpg', frame)
    f += 1


print("done")