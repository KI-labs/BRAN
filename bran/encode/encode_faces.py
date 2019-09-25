# USAGE
# When encoding on laptop, desktop, or GPU (slower, more accurate):
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method cnn
# When encoding on Raspberry Pi (faster, more accurate):
# python encode_faces.py --dataset dataset --encodings encodings.pickle --detection-method hog

# import the necessary packages

# It loads images from a dataset,
# gets the bounding box of the faces and maps each one of them to a 128-dimensional vector
import imutils
from imutils import paths
import face_recognition
import pickle
import cv2
import os


def encode(dataset, detection_method, encodings_path):

    # grab the paths to the input images in our dataset
    print("[INFO] quantifying faces...")
    image_paths = list(paths.list_images(dataset))
    # initialize the list of known encodings and known names
    known_encodings = []
    known_names = []

    # TODO Multithreading
    # loop over the image paths
    for (i, imagePath) in enumerate(image_paths):
        # extract the person name from the image path
        print("[INFO] processing image {}/{}".format(i + 1, len(image_paths)))
        name = imagePath.split(os.path.sep)[-2]

        # load the input image and convert it from RGB (OpenCV ordering)
        # to dlib ordering (RGB)

        image = cv2.imread(imagePath)
        image = imutils.resize(image, width=450)
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # detect the (x, y)-coordinates of the bounding boxes
        # corresponding to each face in the input image
        boxes = face_recognition.face_locations(rgb, model=detection_method)

        # compute the facial X for the face
        encodings = face_recognition.face_encodings(rgb, boxes)

        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            known_encodings.append(encoding)
            known_names.append(name)

    # dump the facial encodings + names to disk
    print("[INFO] serializing encodings...")
    data = {"encodings": known_encodings, "names": known_names}
    f = open(encodings_path, "wb")
    f.write(pickle.dumps(data))
    f.close()
