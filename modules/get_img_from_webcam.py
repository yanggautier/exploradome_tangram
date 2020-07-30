"""
Takes a shot every second and splits it in two.

To quit camera mode, press ESC
"""

import cv2
import os
import time
import shutil

cam = cv2.VideoCapture(0)

cv2.namedWindow("Camera Shot")

img_counter = 0

if not os.path.exists('frames/'):
    os.makedirs('frames/')
else:
    shutil.rmtree('frames/')
    os.makedirs('frames/')

while True:
    ret, frame = cam.read()
    if not ret:
        print("failed to grab frame")
        break
    cv2.imshow("Camera Shot", frame)

    height, width, dim = frame.shape
    width_cutoff = width // 2
    s1 = frame[:, :width_cutoff]
    s2 = frame[:, width_cutoff:]

    #Takes a shot every second
    img_name_A = "frames/frame_{}-A.jpg".format(img_counter)
    img_name_B = "frames/frame_{}-B.jpg".format(img_counter)
    cv2.imwrite(img_name_A, s1)
    cv2.imwrite(img_name_B, s2)
    print("{} written!".format(img_name_A.replace("-A.jpg","")))
    
    img_counter += 1
    time.sleep(1)

    k = cv2.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
        

cam.release()

cv2.destroyAllWindows()

shutil.make_archive('images', 'zip', 'frames/')