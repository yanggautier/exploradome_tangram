# Exploradome_tangram
Tangram form detection from live video stream


## Table Of Contents
-  [Installation and Usage](#Installation-and-Usage)
-  [Usage](#Usage)
-  [Configuration](#Configuration)
-  [In Details](#in-details)
-  [Team](#Team)

## Installation and Usage

- [Tensorflow](https://www.tensorflow.org/) (An open source deep learning platform) 
- [OpenCV](https://opencv.org/) (Open Computer Vision Library)

```bash
pip install opencv-python tensorflow
```

## Approach taken

Find the best accuracy for the model (CNN with Tensorflow) - see the Google Sheet

## In progress

Tested so far:
* MobileNet
* InceptionV3

## Results or improvement strategy

see the Google Sheet: https://docs.google.com/spreadsheets/d/1_P0LEN9CyY8Zfk653IVwfmMUg0E6tyfjU2sLSH3ChIc/edit?usp=sharing

## Configuration

## In Details
```
├──  data  
│    └── data  - here's the image classification datasets (from video_to_img).
│    └── train - here's the file to train dataset.
│    └── validation  		 - here's the file to validation dataset.
│    └── video_to_img    - here's the file of raw image extraction of video file.
│    └── WIN_20200727_16_30_12_Pro.mp4    - here's the tangram video for the creation of the datasets.
│
│
├──  modules        - this file contains the modules.
│    └── get_img_from_webcam.py  - here's the file to extract images of video cam, split in two.
│ 
│
├── saved_model     - this folder contains any customed layers of your project.
│   └── 
│   └──
│
│ 
├── collab Notebooks - this folder contains any model and preprocessing of your project.
│   └── trigram_decoupage.ipynb
│   └── trigram_model_test_Bintou_Jasmine.ipynb
│   └── trigram_model_test_Laura.ipynb
│   └── trigram_model_test_Shadi.ipynb
│   └── trigram_model_test_Jason.ipynb
│   └── trigram_preprocessing.ipynb
│   └── video_processing.ipynb
│   
└── main.py					- this foler contains unit test of your project.
```
### Dataset
![image](https://drive.google.com/uc?export=view&id=1O_vfKNLHZ7HEEBNUZfEWRGjRe7QnCtsS)

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
├──  validation  
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
To download the file:
- data: [data zip](https://drive.google.com/file/d/1Eavrsk72iZeJFiv3NBnOkDxgs2WDF4Ow/view?usp=sharing)
- train: [train zip](https://drive.google.com/file/d/1ZjOI81YRjdcNwF8i6gxLMC1UG02nu2QS/view?usp=sharing)
- validation: [validation zip](https://drive.google.com/file/d/1oCzg1-qK1jKki0bnahJYu7XU5tUbTB_b/view?usp=sharing)
- video_to_img (with all image of video file): [video to img zip](https://drive.google.com/file/d/13XPugnAZIxIP25GGkvKFjiDym-c7eMGL/view?usp=sharing)

### Trigram Preprocessing

### Trigram Model

To use the model, open a new terminal and copy this link:

```
wget -O model.h5 'https://drive.google.com/uc?export=download&id=13dDtd4jsCyA6Z4MEPK3RsWDLiCZJvEPc'
```

## Team

- [Jasmine BANCHEREAU](https://github.com/BeeJasmine)
- [Shadi BOOMI](https://github.com/sboomi)
- [Jason ENGUEHARD](https://github.com/jenguehard)
- [Bintou KOITA](https://github.com/bintou579)
- [Laura TAING](https://github.com/TAINGL)
