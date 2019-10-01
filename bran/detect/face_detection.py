import dlib
import pickle
import time

import cv2
import numpy as np
import face_recognition
from imutils.video import VideoStream, FPS
from sklearn.preprocessing import LabelEncoder

from ..models.distances import DistanceVoting, DistanceMinAvg
from ..blink.blink_detection import update_and_verify_blink_pattern, update_and_verify_smile_pattern, KeyPattern
import requests

WAIT_SECS = 10
HODOR_API = "<---API Endpoint--->"
RESOLUTION = (320, 240)
SIMILARITY_THRESHOLD = 0.85


def draw_rectangles(boxes, frame, names):
    # loop over the recognized faces
    for ((top, right, bottom, left), name) in zip(boxes, names):
        # draw the predicted face name on the image
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        y = top - 15 if top - 15 > 15 else top + 15
        cv2.putText(frame, name, (left, y), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)


def init_model(model_name, path_model, face2vec, confidence):
    if model_name == 'dist_vote':
        cls = DistanceVoting(face2vec, confidence)
    elif model_name == 'dist_avg':
        cls = DistanceMinAvg(face2vec, confidence)
    elif model_name == 'custom':
        with open(path_model, 'rb') as f:
            cls = pickle.load(f)
    else:
        cls = DistanceVoting(face2vec, confidence)
    return cls


def check_mouth_shape_change(shapes, flag):
    """
    :param shapes: mouth_shapes
    :param flag: mouth_change_flag
    :return:
        mouth_change_flag: Set to true when mouth shape pattern changes considerably
    """
    if shapes[0][2] != shapes[1][2] and shapes[1][2] != shapes[2][2] \
            and shapes[0][3] != shapes[1][3] and shapes[1][3] != shapes[2][3]:
        flag = True

    return flag


def detect(cascade, encodings, shape_predictor, model_name, confidence, path_model, pi_camera, liveliness_type,
           liveliness_pattern):

    # Blink and or Smile detection included
    # initialize the frame counters and the total number of blinks

    # THE SEQUENCE SHOULD BE DEFINED MANUALLY
    liveliness_pattern = list(liveliness_pattern)

    unique = np.unique(liveliness_pattern)
    assert 'x' in unique and 'o' in unique

    idx = np.array(liveliness_pattern) == 'x'
    key = np.zeros(shape=(len(liveliness_pattern),))
    key[idx] = 1

    key_pattern = KeyPattern(key, 20)

    # load the known faces and embeddings along with OpenCV's Haar
    # cascade for face detection

    print("[INFO] loading encodings dataset...")
    dataset = pickle.loads(open(encodings, "rb").read())

    le = LabelEncoder()
    dataset['idx'] = le.fit_transform(dataset['names'])
    classes = np.array([c for c in le.classes_] + ['Unknown'])

    print("[INFO] init model...")
    cls = init_model(model_name, path_model, dataset, confidence)

    print("[INFO] loading face detector...")
    detector = cv2.CascadeClassifier(cascade)

    cv2.namedWindow("frame", cv2.WINDOW_NORMAL)
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")
    if pi_camera:
        vs = VideoStream(usePiCamera=True).start()
    else:
        vs = VideoStream(src=0, resolution=RESOLUTION).start()

    time.sleep(2.0)

    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    print("[INFO] loading facial landmark predictor...")
    print("[INFO] loading facial landmark predictor...")
    face_shape_predictor = dlib.shape_predictor(shape_predictor)

    # Adding mouth_shapes array
    print("Starting to collect mouth shapes")
    mouth_shapes = []
    smile_flag = False
    # start the FPS counter
    fps = FPS().start()

    while True:

        # grab the frame from the threaded video stream and resize it
        # to 500px (to speedup processing)
        frame = vs.read()

        # convert the input frame from
        # 1) BGR to grayscale (for face detection)
        # 2) from BGR to RGB (for face recognition)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # detect faces in the grayscale frame
        rects = detector.detectMultiScale(gray,
                                          scaleFactor=1.1,
                                          minNeighbors=5,
                                          minSize=(30, 30),
                                          flags=cv2.CASCADE_SCALE_IMAGE)

        if len(rects) > 0:

            # keep only the rectangle with the biggest volume
            rect_sizes = np.array(map(lambda x: x[2] * x[3], rects))

            rect = rects[rect_sizes.argmax()]

            if liveliness_type == "smile":
                similarity, max_similarity, mouth_shapes, smile_flag = update_and_verify_smile_pattern(face_shape_predictor, frame, gray, rect, key_pattern, mouth_shapes, smile_flag)
            elif liveliness_type == "blink":
                similarity, max_similarity = update_and_verify_blink_pattern(face_shape_predictor, frame, gray, rect, key_pattern)
            else:
                raise ValueError('liveliness type not valid')

            # OpenCV returns bounding box coordinates in (x, y, w, h) order
            # but we need them in (top, right, bottom, left) order, so we
            # need to do a bit of reordering
            (x, y, w, h) = rect
            boxes = [(y, x + w, y + h, x)]

            # compute the facial embeddings for each face bounding box
            encodings = face_recognition.face_encodings(rgb, boxes)

            names = []
            if encodings and model_name == 'custom':
                names = custom_prediction(classes, cls, confidence, encodings, names)
            elif encodings:
                names = list(classes[cls.predict(encodings).astype(int)])

            draw_rectangles(boxes, frame, names)

            if not smile_flag:
                print("Please smile to enter")

            if smile_flag:
                mouth_shapes_after_smile = mouth_shapes

            if smile_flag and mouth_shapes_after_smile[-1] < 0.1 and names[-1] != 'Unknown':
                    # and pi_camera:
                mouth_shapes = []
                mouth_shapes_after_smile = []
                smile_flag = False
                # uncomment the following line to post to an API endpoint if any
                print('INFO - {}'.format("Invoking Hodor"))
                # uncomment the following line to post to an API endpoint if any
                # print('INFO - {}'.format(requests.post(HODOR_API, json={})))
                time.sleep(WAIT_SECS)
                key_pattern.clean_memory()

        # display the image to our screen
        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

        cv2.imshow('frame', frame)

        # update the FPS counter
        fps.update()

    # stop the timer and display FPS information
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

    cv2.destroyAllWindows()
    vs.stop()


def custom_prediction(classes, cls, confidence, encodings, names):
    proba = cls.predict_proba(encodings)
    single_proba = np.max(proba, axis=1)
    idx = np.argmax(proba, axis=1)
    idx[single_proba <= confidence] = -1
    names = list(classes[idx])
    return names
