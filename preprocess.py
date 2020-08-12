from PIL import Image
import cv2
import numpy as np

# PREPROCESS FUNCTION FOR PREDICTION ON A SINGLE FRAME
def preprocess_image(img, side=250, split="none"): 
    '''
    @andres + francis
    input : raw image
    returns processed image for prediction
    split : string
            "none", "right" or "left" to crop input image
    '''
    if split == "left" :
        img = img[:img.shape[1]//2, :img.shape[1]//2]
    elif split == "right":
        img = img[:img.shape[1]//2, img.shape[1]//2 :]
    else:
        pass

    # RESIZE
    img = cv2.resize(img, (side,side))
    img_test = img.copy()
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img / 255.0
    #img = np.expand_dims(img, axis=0)
    img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))
    return img, img_test