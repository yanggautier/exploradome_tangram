# TangrIAm Project

The project is partnership between Exploradôme museum, OCTO Technology and Microsoft and it aims to introduce the concept and application of artificial intelligence to young children. The specific application developed for the project is to apply object detection to live tangram solving.

A tangram is a dissection puzzle consisting of seven flat polygons (5 triangles, 1 square and 1 parallelogram) which are combined to obtain a specific shape. The objective is to replicate a pattern (given only an outline) using all seven pieces without overlap.

Within the framework of the project, 12 tangram selected shapes act as classes for the object detector:

![image](https://drive.google.com/uc?export=view&id=1O_vfKNLHZ7HEEBNUZfEWRGjRe7QnCtsS)

boat(bateau), bowl(bol), cat(chat), heart(coeur), swan(cygne), rabbit(lapin), house(maison), hammer(marteau), mountain(montagne), bridge(pont), fox(renard), turtle(tortue)

## Objective

Classify tangram shapes from a live video stream using transfer learning as the main basis of our model.

## Table Of Contents
-  [Installation and Usage](#Installation-and-Usage)
-  [Dataset Creation](#Dataset-Creation)
-  [Model Creation](#Model-Creation)
  -  [Transfer learning](#Transfer-learning)
-  [Getting Started](#Getting-Started)
-  [Command Line Args Reference](#Command-Line-Args-Reference)
-  [References](#References)
-  [Team](#Team)

# Dataset Creation

## 1. Video recording
To create the dataset our image classification, we need to have images with label of each category of tangram.
To do this, we filmed continuously members of our team performing in turn the 12 shapes possibles, by using the camera provided by Exploradome to respect the conditions under which the algorithm will be used.

## 2. Image dataset preparation

To prepare the dataset, we needed to sample images of each shape from a video.
* We sampled 1 image/second
* We divided each image in half to get more samples
* We manually selected the ones where the shape was distinguishable enough

## 3. Images labeling
TensorFlow requires the dataset to be provided in the following directory structure:
Like this, that why each photo is order in folder with the name of category :
```
├──  multilabel_data  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│ 
│ 
├── 
```
We have already created the dataset in this format and provided a download link (and some instructions) in the GitHub repository. 

## 4. Initial Dataset

The initial dataset is unbalanced between categoriy. 
We didn't split already the dataset between training data and testing before applying data augmentation.

| Label           |  Total images | 
|-----------------|------|
|boat(bateau)     | 716  | 
| bowl(bol)       | 248  |  
| cat(chat)       | 266  | 
| heart(coeur)    | 273  |  
| swan(cygne)     | 321  |  
| rabbit(lapin)   | 257  |  
| house(maison)   | 456  |  
| hammer(marteau) | 403  |  
| mountain(montagne)  |  573 |  
| bridge(pont)    | 709  |  
| fox(renard)     | 768  |  
| turtle(tortue)  | 314  |  
| TOTAL           | 5304 | 

## 5. Data augmentation
Having a large dataset is crucial for the performance of the deep learning model.
Data augmentation is a strategy to increase the diversity of data available for training models, without actually collecting new data.

For our dataset we applied different types images augmentations to obtain more images.

Data Augmentation with python scripts:
- Contrast changes (1.5 #brightens the image) with PIL and ImageEnhance with `Brightness()`
- Blurring (applied after contrast change) with OpenCV and cv2 with `gaussianblur()` 

`ImageDataGenerator` with TensorFlow:
* Rescaling : 1./255 is to transform every pixel value from range [0,255] -> [0,1]
* Rotation : each picture is rotated with a random angle from 0° to 90°
* Flipping : each picture gets flipped on both axis (vertical and horizontal)
* Split train_full or train_balanced dataset to train and validation dataset (= 30% of train dataset)

```python
image_gen_train = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=90,
                    horizontal_flip=True,
                    vertical_flip=True,
                    validation_split=0.3)
```

| Label           |  Before Data Augmentation  |   After Data Augmentation* | 
|-----------------|---------------|----------------|
| boat(bateau)    | 716           |   2148         | 
| bowl(bol)       | 248           |   744          | 
| cat(chat)       | 266           |   800          | 
| heart(coeur)    | 273           |   820          | 
| swan(cygne)     | 321           |   964          | 
| rabbit(lapin)   | 257           |   772          | 
| house(maison)   | 456           |   1368         | 
| hammer(marteau) | 403           |   1209         | 
| mountain(montagne)  |  573      |   1720         | 
| bridge(pont)    | 709           |   2128         | 
| fox(renard)     | 768           |   2304         |  
| turtle(tortue)  | 314           |   942          |  
| **TOTAL**           | **5304**          |   **15919**        | 

* with script python

Next step, we created a balanced datasets. 
For each category we keep randomly:
- 400 images for the training dataset 
- 80 images (20% of 400) for the test dataset 

The dataset has the following directory structure:
```
├──  train  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│ 
│ 
├──  test  
│    └── bateau: [bateau.1.jpg, bateau.2.jpg, bateau.3.jpg ....]  
│    └── bol: [bol.1.jpg, bol.2.jpg, bom.3.jpg ....]    
│    └── chat  ... 		   
│    └── coeur ...  
│    └── cygne ...
│    └── lapin ...
│    └── maison ...
│    └── marteau ...
│    └── montagne ...
│    └── pont ...
│    └── renard ...
│    └── tortue ...
│   
└── 
```

To access the dataset, visit the [Google Drive link](https://drive.google.com/drive/folders/1LQO_zfVZ-niiVsCqzQEUEZHry8aATK2s?usp=sharing). The folder contains both training and validation sets.

# Model Creation
## Transfer learning
**What is Transfer Learning?**
Transfer learning is a machine learning technique in which a network that has already been trained to perform a specific task is repurposed as a starting point for another similar task. 

**Transfer Learning Strategies & Advantages:**
There two transfer learning strategies, here we use:
   * Initialize the CNN network with the pre-trained weights
   * We then retrain the entire CNN network while setting the learning rate to be very small, which ensures that we don't drastically change the trained weights
   
The advantage of transfer learning is that it provides fast training progress since we're not starting from scratch. Transfer learning is also very useful when you have a small training dataset available, but there's a large dataset in a similar domain (i.e. ImageNet).

**Using Pretrained Model:**
There are 2 ways to create models in Keras. Here we used the sequential model.
The sequential model is a linear stack of layers. You can simply keep adding layers in a sequential model just by calling add method. 

The two pretrained models used are: 
* [MobileNetV2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/MobileNetV2): lightweight, used for laptops
* [InceptionV3 + L2](https://www.tensorflow.org/api_docs/python/tf/keras/applications/InceptionV3): heavier, used for image analysis

**Transfer Learning with Image Data**
It is common to perform transfer learning with predictive modeling problems that use image data as input.

This may be a prediction task that takes photographs or video data as input.

For these types of problems, it is common to use a deep learning model pre-trained for a large and challenging image classification task such as the [ImageNet](http://www.image-net.org/) 1000-class photograph classification competition.

These models can be downloaded and incorporated directly into new models that expect image data as input.

## Apply Transfer Learning

```python
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
```

```python
inception = InceptionV3(weights='imagenet', include_top=False)
```

## Results or improvement strategy

See the Google Sheet: https://docs.google.com/spreadsheets/d/1_P0LEN9CyY8Zfk653IVwfmMUg0E6tyfjU2sLSH3ChIc/edit?usp=sharing

# Getting Started

## In Details

* The Notebook section: a detailled explanation on how models are created
* `model` folder: where models can be stocked
* `modules` folder: where several python scripts can be stocked for experimental purposes
* `test_tangram.py` the main file used to launch the identification

## Installation and Usage

- [Tensorflow](https://www.tensorflow.org/) (An open source deep learning platform) 
- [OpenCV](https://opencv.org/) (Open Computer Vision Library)
- Python 3.7.x, 64bit

```bash
pip install opencv-python tensorflow
```

## Get more models

Get inside the model folder :

```
cd models/
```

**Inception V3**

```
wget -O InceptionV3.h5 https://drive.google.com/uc?export=download&id=1G2dIFlRW2IVDehxZAU6kU72Ps9YMvzaI
```

**MobileNetV2**

```
wget -O MobileNetV2.h5 https://drive.google.com/uc?export=download&id=13dDtd4jsCyA6Z4MEPK3RsWDLiCZJvEPc
```
## Inference

All model files can be found in the models folder. To use a model for inference, either connect the camera to your device or select a video file and write the following command line:

```
python test_tangram -c [camera] -s [side : left | right] -o [output_folder] -m [model] -i [input folder (OPTIONAL)]
```

**Example:**

```
python test_tangram.py -c 1 -s left -o result_pics -m models\tangram_jason_mobilenet_final_06082020.h5
```


# Team

- [Jasmine BANCHEREAU](https://github.com/BeeJasmine)
- [Shadi BOOMI](https://github.com/sboomi)
- [Jason ENGUEHARD](https://github.com/jenguehard)
- [Bintou KOITA](https://github.com/bintou579)
- [Laura TAING](https://github.com/TAINGL)