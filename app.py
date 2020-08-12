import cv2
import numpy as np
import time
import tensorflow as tf
import pandas as pd
from PIL import Image
from keras.models import load_model

# Calls preprocess_image function from preprocess.py 
from preprocess import preprocess_image

# import model
model = tf.keras.models.load_model("model_gridsearch.h5")

# Labels
Labels = ['bateau', 'bol', 'chat', 'coeur', 'cygne', 'lapin', 'maison', 'marteau', 'montagne', 'pont', 'renard','tortue'] # Labels

# Number of predictions to display :
nb_display = 5

# CUSTOMIZATION OF FONTS
font = cv2.FONT_HERSHEY_SIMPLEX # font
fontScale = 1   # fontScale
color = (255, 0, 0)    # Blue color in BGR
thickness = 2  # Line thickness of 2 px 

# open camera or video
cap = cv2.VideoCapture(0)

# Nom de la fenetre
cv2.namedWindow("Exploradome", cv2.WINDOW_NORMAL)

# initialise predictions text
text = ''

while True:

    # Read frame from camera
    ret, frame=cap.read()

    # Loop break if camera problem
    if not ret: 
        print("failed to grab frame")
        break

    # Display continuously    
    k = cv2.waitKey(1)
    
    # CALL PREPROCESS FUNCTION IN preprocess.py
    img, img_test = preprocess_image(frame, side=250, split='None')

    # Predict
    preds = model.predict(img)

    # Pedictions to text to superimpose
    text = f"{Labels[np.argmax(preds)]} {round(max(preds[0]), 2) *100}"
    df = pd.DataFrame({"label": Labels, "predictions":preds[0]})
    df = df.sort_values("predictions", axis=0, ascending=False, ignore_index=True)
    for i in range(nb_display):
        temp = df["label"][i] + " " + str(round(df["predictions"][i]*100,2)) + " %" 
        cv2.putText(frame, temp, (20, i*50+40), font, fontScale, color, thickness, cv2.LINE_AA)

    # display image
    cv2.imshow('Camera', frame)

    # key "q" to quit
    if cv2.waitKey(1)&0xFF==ord('q'):
        break

cap.release()  # Laisse la caméra libre
cv2.destroyAllWindows()  # ferme fenêtres