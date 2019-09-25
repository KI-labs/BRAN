
# BRAN (Basic Recognition and Authorisation at eNtrance)

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
#### Custom Model (e.g. Multi-layer perceptron aka Deep Neural Network)
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
#### Custom Model (e.g. Multi-layer perceptron aka Deep Neural Network)  
````bash
bran detect -c haarcascade_frontalface_default.xml -e encodings.pickle -p shape_predictor_68_face_landmarks.dat -m custom -k models/mlp.pickle -t 0.9
````
