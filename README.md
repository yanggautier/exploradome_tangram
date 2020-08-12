# **TangrIAm** Project

The project is a partnership between Exploradôme museum, OCTO Technology and Microsoft and it aims to introduce the concept and application of artificial intelligence to young children. The specific application developed for the project is to apply object detection to live tangram solving.

A tangram is a dissection puzzle consisting of seven flat polygons (5 triangles, 1 square and 1 parallelogram) which are combined to obtain a specific shape. The objective is to replicate a pattern (given only an outline) using all seven pieces without overlap.

Within the framework of the project, 12 tangram selected shapes act as classes for the classifier:

-  boat (bateau)

- bowl (bol)

-  bridge (pont)

-  cat (chat)

-  fox (renard)

- hammer  (marteau)

-  heart (coeur)

-   house (maison)

-  mountain (montagne)

- rabbit (lapin)

-  swan (cygne)

- turtle (tortue)




**Objective :**

The objective of the project is to train a custom Convolutional Neural Network (CNN) model to perform real-time recognition of tangram shapes.

The model is built using Keras API on a TensorFlow backend. 

## Data

### **Data collection**

- Training data was collected by taking a sample video with the webcam (to be used for live recording) and breaking it down into frames (images). 

  The breakdown into frames was made with VLC Media Player.

  Link to video [here](https://drive.google.com/file/d/1bX_x2rNIOm3q86X5xBEyLZxVzltYR2bD/view?usp=sharing)

  Each resulting image was cut in half to obtain two images with a tangram shape on both sides of the board, using :



  ```python
  from PIL import ImageFile
  ImageFile.LOAD_TRUNCATED_IMAGES = True
  
  import os
  import imageio
  count = 0
  classe = "cygnes"
  for root, dirs, files in os.walk(f"C:/Users/ouizb/OneDrive/Pictures/Exploradrome_image/{classe}", topdown = False):
      for name in files:
          os.path.join(root, name)
          image = imageio.imread(os.path.join(root, name))
          height, width = image.shape[:2]
          width_cutoff = width // 2
          s1 = image[:, :width_cutoff]
          s2 = image[:, width_cutoff : ]
          status = imageio.imwrite(f'C:/Users/ouizb/OneDrive/Pictures/Exploradrome_image/image_coupe/{classe}/{classe}_left_{count}.jpg', s1)
          print("Image written to file-system : ",status)
          status = imageio.imwrite(f'C:/Users/ouizb/OneDrive/Pictures/Exploradrome_image/image_coupe/{classe}/{classe}_right_{count}.jpg', s2)
          print("Image written to file-system : ",status)
          count += 1
  ```



  <img src="C:\Users\ouizb\OneDrive\Pictures\Exploradrome_image\Data initial\Bateau\capture00002.png" alt="capture00002" style="zoom:25%;" />







  <img src="https://drive.google.com/uc?id=1dAgHbEwZXp-6DNwXGkc_up52oMJuMe55" alt="image" style="zoom:25%;" />



  <img src="https://drive.google.com/uc?id=1g5jz2DhgeQWMO9unGYw34ncxnddanQlY" alt="image" style="zoom: 25%;" />



  The resulting images were saved in 12 separate folders (by class). 

  Only images with no foreign object (e.g. hands) obstructing the tangram shape were retained. 

  The resulting dataset aimed to relatively balance the available training images by class. The following is the number of initial images per class. 

  - ​	boat: 246 images
  - ​	bowl: 148 images
  - ​	cat: 100 images
  - ​	heart: 216 images
  - ​	swan: 248 images
  - ​	rabbit: 247 images
  - ​	house: 136 images
  - ​	hammer: 245 images
  - ​	mountain: 313 images
  - ​	bridge: 431 images
  - ​	fox: 502 images
  - ​	turtle: 164 images

  The dataset is available [here](https://drive.google.com/drive/folders/1CK7x1mHU27PEGIR34WgxyCYxj0yGd9lz?usp=sharing)



  ### **Data augmentation**

  The dataset was further augmented and split into training (70% of data), validation (20% of data) and test (10% of data) using [Roboflow](https://roboflow.ai/).

  After data augmentation, each class had 1140 images in the training dataset.

  The filter used to augment the data were:

  - ​	rotation 15° of each side
  - ​	shear
  - ​	brightness
  - ​	blur
  - ​	noise

  The dataset is available [here](https://drive.google.com/drive/folders/1VSARFx8Y8r9yEGKA9lutmm-34AHzeS51?usp=sharing)

# Model

The model was first trained on the initial dataset and despite a good performance on the training and validation set, the model failed to generalize well when tested on a live video stream.  

The following table records the model performance on the initial data. 

Further on, the model was trained on the augment data. 

The model performance on the augmented dataset is presented [here](https://simplonformations-my.sharepoint.com/:x:/g/personal/fmujani_simplonformations_onmicrosoft_com/EYL8EaznSh5LvV0Jm_7D3ekB7MfpqFQv99vXPj7SP2V8Jw?e=mJAaT4)

Link to models [here](https://drive.google.com/drive/u/1/folders/1GpLE5O6VSEYY6Wemhw5pcsaNKVeQSVCq)



# Inference

```python
Labels = ['bateau', 'bol', 'chat', 'coeur', 'cygne', 'lapin', 'maison', 'marteau', 'montagne', 'pont', 'renard','tortue']
# cam = cv2.VideoCapture('C:/Users/utilisateur/Dropbox/SIMPLON/eploradom/data/WIN_20200727_16_07_25_Pro.mp4')
cam = cv2.VideoCapture(1) # 0: cam, path pour video
cv2.namedWindow("test")
# font 
font = cv2.FONT_HERSHEY_SIMPLEX  
# fontScale 
fontScale = 1  
# Blue color in BGR 
color = (255, 0, 0)   
# Line thickness of 2 px 
thickness = 2
count = 0
start = time.time()
nb_display = 5
while True:
    text = ''
    ret, frame = cam.read() # lancer la capture. ret=True si tout ok.
    if not ret: # si problème : casse la boucle
        print("failed to grab frame")
        break
        
    
    k = cv2.waitKey(1)  # waitkey(n) dans une boucle affiche frame par frame. n = n milliseconds
    
    if k%256 == 27:     # si ESC pressé : casse la boucle
        print("Escape hit, closing...")
        break
    
#     elif k%256 == 32: # si espace pressé :
#     elif count % 8 == 0 :
    img, img_test = preprocess_image(frame, side=250, split="right")
    
    preds = model.predict(img)
    df = pd.DataFrame({"label": Labels, "predictions":preds[0]})
    df = df.sort_values("predictions", axis=0, ascending=False, ignore_index=True)
    for i in range(nb_display):
        temp = df["label"][i] + " " + str(round(df["predictions"][i]*100,2)) + " %" 
        cv2.putText(frame, temp, (20, i*50+40), font, fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow("test", frame) # affiche l'image dans fenêtre nomée "test"
    count += 1
    
    if count % 10 == 0 :
        cv2.imwrite(f"./bin/renard/img_{count}.jpg", img_test) # Test : Sur quelle image il prédit ??? !!!
        
end = time.time() 
duree = end - start
print(count)
print(duree)
print(count/duree)
cam.release()  # Laisse la caméra libre
cv2.destroyAllWindows()  # ferme fenêtres
```
