"""
Takes a shot every second and splits it in two. Saves the result in a 
folder called 'frames'.

To quit camera mode, press ESC
"""

import cv2
import os
import time
import shutil
import tensorflow as tf

CLASS_NAMES = ['maison', 'marteau', 'tortue', 'montagne', 
'chat', 'bol', 'coeur', 'pont', 'lapin', 'bateau', 'renard', 'cygne']

# Must import model.h5 as model

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

    # Resize image to expected size for the model and expansion of dimension from 3 to 4
    s1_up = tf.image.resize(s1, (224,224), preserve_aspect_ratio=False)
    s1_final = tf.expand_dims(s1_up, axis=0)
    s2_up = tf.image.resize(s2, (224,224), preserve_aspect_ratio=False)
    s2_final = tf.expand_dims(s2_up, axis=0)
    
    # Prediction and creation of results dictionnaries
    result_1 = model.predict(s1_final)
    result_dict_1 = {}
    result_2 = model.predict(s2_final)
    result_dict_2 = {}
    for i in range(result_1.shape[1]):
        result_dict_1[CLASS_NAMES[i]] = (str(result_1[0][i])+" %")
        result_dict_2[CLASS_NAMES[i]] = (str(result_2[0][i])+" %")
    print(result_dict_1)
    print(result_dict_2)

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