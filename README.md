
# BRAN

[![GitHub license](https://img.shields.io/badge/license-bsd-green.svg)](https://github.com/KI-labs/BRAN/master/LICENSE)

**B**asic **R**ecognition and **A**uthentication at e**N**trance

A Facial recognition based identification & authentication system at [KI labs](https://ki-labs.com/).

![alt text](assets/logo.png "BRAN")

## Index

- [About](#about)
- [Introduction](#introduction)
- [Installation](#installation)
  - [Commands](#commands)
- [Directory Structure](#file-structure)
-  [Usage](#usage)
  - [Encode Faces](#usage)
    - [Hog method](#usage)
    - [CNN method](#usage)
  - [Detect Faces - PI mode](#usage)
    - [Euclidean Distance - Maximum Votes](#usage)
    - [Euclidean Distance - min(Average Euclidean Distance)](#usage)
    - [Custom Model (Multi-layer perceptrons/DNNs)](#usage)
  - [Detect Faces - Local mode](#usage)
    - [Euclidean Distance - Maximum Votes](#usage)
    - [Euclidean Distance - min(Average Euclidean Distance)](#usage)
    - [Custom Model (Multi-layer perceptrons/DNNs)](#usage)
- [Contribution](#contribution)

###  Introduction

*BRAN* is an identification and authentication system mounted at the entrance of our cool [KI labs](https://ki-labs.com/) office, here in Munich. 

BRAN uses some of the popular facial recognition algorithms in order to identify and allow authorized personnel into our office premises. 

BRAN in essence, consists of these components:

- Raspberry Pi Model 3B+
- Basic Pi Camera Module (5MP) - to stream in the video captured at the entrance
- OpenCV based facial recognition
- **[Hodor]**(https://medium.com/ki-labs-engineering/hodor-controlling-the-office-door-from-slack-a79e77635e39) - Our existing door opening application

BRAN is also our first attempt at understanding and evaluating simple facial recognition algorithms such as Haar Cascade Classifiers to more complex ones such as HOG, Linear SVM and CNNs.

### Installation

- Clone this repository

Run the following commands to:
 
- To navigate the root of the repository  
- Install all dependent libraries and packages

## Commands

```
$ cd BRAN
```

```
$ make setup
```

###  Directory Structure
Add a file structure here with the basic details about files, below is an example.

```
.
├── Makefile
├── README.md
├── assets
│   └── logo.png
├── bran
│   ├── __init__.py
│   ├── __main__.py
│   ├── blink
│   │   ├── __init__.py
│   │   └── blink_detection.py
│   ├── cross_validation
│   │   ├── __init__.py
│   │   ├── cross_val.py
│   │   └── plot.py
│   ├── detect
│   │   ├── __init__.py
│   │   └── face_detection.py
│   ├── encode
│   │   ├── __init__.py
│   │   └── encode_faces.py
│   └── models
│       ├── __init__.py
│       ├── distances.py
│       └── train_and_save_custom.py
├── dataset
│   └── M\ S\ Shankar
├── encodings.pickle
├── haarcascade_frontalface_default.xml
├── models
│   ├── log.pickle
│   └── mlp.pickle
├── requirements.txt
├── setup.py
└── shape_predictor_68_face_landmarks.dat
```

## Usage

### Encode Faces
#### Hog method
````bash
bran encode --dataset dataset --encodings encodings.pickle --detection-method hog
````
#### CNN method
````bash
bran encode --dataset dataset --encodings encodings.pickle --detection-method cnn
````

### Detect Faces - PI Mode
#### Euclidean Distance - Maximum Votes
````bash
bran detect -c haarcascade_frontalface_default.xml -e encodings.pickle -p shape_predictor_68_face_landmarks.dat -m dist_vote -t 0.5 -f
````
#### Euclidean Distance - min(Average Euclidean Distance)
````bash
bran detect -c haarcascade_frontalface_default.xml -e encodings.pickle -p shape_predictor_68_face_landmarks.dat -m dist_avg -t 0.5 -f
````
#### Custom Model (Multi-layer perceptrons/ DNNs)
````bash
bran detect -c haarcascade_frontalface_default.xml -e encodings.pickle -p shape_predictor_68_face_landmarks.dat -m custom -k models/mlp.pickle -t 0.9 -f
````
### Detect Faces - Local Mode
#### Euclidean Distance - Maximum Votes
````bash
bran detect -c haarcascade_frontalface_default.xml -e encodings.pickle -p shape_predictor_68_face_landmarks.dat -m dist_vote -t 0.5
````
#### Euclidean Distance - min(Average Euclidean Distance)
````bash
bran detect -c haarcascade_frontalface_default.xml -e encodings.pickle -p shape_predictor_68_face_landmarks.dat -m dist_avg -t 0.5
````
#### Custom Model (Multi-layer perceptrons/ DNNs)  
````bash
bran detect -c haarcascade_frontalface_default.xml -e encodings.pickle -p shape_predictor_68_face_landmarks.dat -m custom -k models/mlp.pickle -t 0.9
````


### Contribution

 **BRAN** is only a hackathon project and a brief glimpse of what could be achieved using some of the interesting concepts and algorithms in computer vision.

 Therefore, any sort of suggestions, feedbacks and contributions are also always welcome and appreciated. 
 
 Please take note of the following things if you wish to contribute to this project.

 1. **Report a bug** <br>
 If you think you have encountered a bug, and we should know about it, feel free to report it [here](https://github.com/KI-labs/BRAN/issues).

 2. **Request a feature** <br>
 You could also request for a feature [here](https://github.com/KI-labs/BRAN/issues).  

 3. **Create a pull request** <br>
 It can't get better then this, your pull request will be appreciated by the community. You can get started by picking up any open issues from [here](https://github.com/KI-labs/BRAN/issues) and make a pull request.
