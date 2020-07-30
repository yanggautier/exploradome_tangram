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
pip install 
```

## Approach taken

Find the best accuracy for the model (CNN with Tensorflow) - see the Google Sheet

## In progress

Test different model: CNN with Tensorflow 

## Results or improvement strategy

see the Google Sheet: https://docs.google.com/spreadsheets/d/1_P0LEN9CyY8Zfk653IVwfmMUg0E6tyfjU2sLSH3ChIc/edit?usp=sharing

## Configuration

## In Details
```
├──  data  
│    └── test  - here's the image classification datasets (from video_to_img).
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
├── saved_model      - this folder contains any customed layers of your project.
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

```

├──  validation  
│    └── bateau  
│    └── bol  
│    └── chat  		   
│    └── coeur   
│    └── cygne
│    └── lapin
│    └── maison
│    └── marteau
│    └── montagne
│    └── pont
│    └── renard
│    └── tortue
│   
└──
```

### Trigram Preprocessing

### Trigram Model

## Team

- [Jasmine BANCHEREAU](https://github.com/BeeJasmine)
- [Shadi BOOMI](https://github.com/sboomi)
- [Jason ENGUEHARD]()
- [Bintou KOITA](https://github.com/bintou579)
- [Laura TAING](https://github.com/TAINGL)
