# import the necessary packages
import cv2
import numpy as np
# Blink_detection included
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib


class KeyPattern(object):
    """Check if a recorded pattern of zeros and ones
    matches with a predefined key.
    """

    def __init__(self, key, memory=2):
        """
        :param key:
            pattern of zeros and ones the represents a sequence of blinks or smiles
        :param memory:
            list of last n pattern comparisons
        """
        self.key = key
        self.length = max(len(key), 1)
        self.pattern = np.zeros(shape=self.length)
        self.similarities = np.zeros(shape=memory)

    def update(self, pattern_new_el):
        """Add the new element in the end of the array,
        drop the oldest (first) element from the array
        """
        self.pattern[:-1] = self.pattern[1:]
        self.pattern[-1] = pattern_new_el
        self.similarities[:-1] = self.similarities[1:]
        self.similarities[-1] = (self.key == self.pattern).sum() / self.length
        return self

    def match(self, threshold=0.8):
        if (self.key == self.pattern).sum() / self.length > threshold:
            return True

        return False

    def similarity(self):
        return self.similarities[-1]

    def max_similarity(self):
        """Maximum similarity from the last `memory` pattern comparisons.

        If `memory`=2 and you have generated a pattern whose similarity with the key is above
        the threshold but at that moment your face was not recognized in the next frame you
        get the chance to use this similarity (since it is stored in the similarities list).

        If `memory`=n you have this chance for the next `n-1` frames.
        """
        return self.similarities.max()

    def clean_memory(self):
        self.pattern = self.pattern * 0
        self.similarities = self.similarities * 0


def mouth_features(shape):
    """
    shape = np.array([[x1,y1], [x2,y2], ..])
    """

    mouth_left_Edge = shape[60]
    mouth_right_Edge = shape[64]

    mouth_top_m = shape[62]
    mouth_bottom_m = shape[66]

    m = max(np.linalg.norm(mouth_top_m - mouth_bottom_m), 1)

    t1 = np.linalg.norm(mouth_left_Edge - mouth_top_m)
    t2 = np.linalg.norm(mouth_right_Edge - mouth_top_m)

    b1 = np.linalg.norm(mouth_left_Edge - mouth_bottom_m)
    b2 = np.linalg.norm(mouth_right_Edge - mouth_bottom_m)

    ee = np.linalg.norm(mouth_left_Edge - mouth_right_Edge)

    rad_to_deg = 57.2958

    alpha_1 = 0
    alpha_2 = 0

    if t1 > 0 and m > 0:
        z = (t1 ** 2 + m ** 2 - b1 ** 2) / (2 * t1 * m)
        if -1 <= z <= 1:
            alpha_1 = np.arccos(z) * rad_to_deg

    if t2 > 0 and m > 0:
        z = (t2 ** 2 + m ** 2 - b2 ** 2) / (2 * t2 * m)
        if -1 <= z <= 1:
            alpha_2 = np.arccos(z) * rad_to_deg

    mouth_ratio = m / ee

    return alpha_1, alpha_2, mouth_ratio, ee


def eye_aspect_ratio(eye):
    """
    :param eye:
    :return: eye aspect ratio:
    """
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])
    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)
    # return the eye aspect ratio
    return ear


def update_and_verify_smile_pattern(face_landmark_predictor, frame, gray, rect, key_pattern, mouth_shapes, smile_flag):
    """
    :param face_landmark_predictor: detect facial landmarks
    :param frame: image in rgb
    :param gray: image in grey
    :param rect: rectangle obtained by appplying cv2.CascadeClassifier().detectMultiScale(..)
    rect = (x, y, w, h)
    :param key_pattern: member of class Key_Pattern
    :return:
        similarity: pattern match in the interval [t-pattern_length, t] with the key
        max_similarity: similarity obtained in a certain time interval
    """
    alpha_min, mouth_ratio_min = 75, 0.3

    (x_, y_, w_, h_) = rect
    rect = dlib.rectangle(int(x_), int(y_), int(x_ + w_), int(y_ + h_))
    shape = face_landmark_predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    alpha_1, alpha_2, mouth_ratio, ee = mouth_features(shape)

    # mouth_shape = [alpha_1, alpha_2, mouth_ratio, ee]
    mouth_shape = mouth_ratio

    mouth_shapes.append(mouth_shape)

    smile = 0
    if alpha_1 > alpha_min and alpha_2 > alpha_min and mouth_ratio > mouth_ratio_min:
        smile = 1
        smile_flag = True

    key_pattern.update(smile)
    similarity = key_pattern.similarity()
    max_similarity = key_pattern.max_similarity()

    for (x, y) in shape[62: 62 + 1]:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    for (x, y) in shape[66: 66 + 1]:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    for (x, y) in shape[60: 60 + 1]:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    for (x, y) in shape[64: 64 + 1]:
        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

    memory_str = np.array2string(key_pattern.pattern, precision=1, separator=',', suppress_small=True)

    cv2.putText(frame, "pattern: {:s}".format(memory_str), (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # cv2.putText(frame, "alpha1: {:.2f}".format(alpha_1), (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.putText(frame, "alpha2: {:.2f}".format(alpha_2), (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.putText(frame, "ratio: {:.2f}".format(mouth_ratio), (260, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.putText(frame, "smile: {:d}".format(smile), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    # cv2.putText(frame, "sim: {:.2f}".format(similarity), (260, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.putText(frame, "maxsim: {:.2f}".format(max_similarity), (240, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255)
    # ,2)

    return similarity, max_similarity, mouth_shapes, smile_flag


def update_and_verify_blink_pattern(face_landmark_predictor, frame, gray, rect, keyPattern):
    """
    :param face_landmark_predictor: detect facial landmarks
    :param frame: image in rgb
    :param gray: image in grey
    :param rect: rectangle obtained by appplying cv2.CascadeClassifier().detectMultiScale(..)
    rect = (x, y, w, h)
    :param keyPattern: member of class Key_Pattern
    :return:
        similarity: pattern match in the interval [t-pattern_length, t] with the key
        max_similarity: similarity obtained in a certain time interval
    """
    eyeClosedThreshold = 0.25

    (x_, y_, w_, h_) = rect
    rect = dlib.rectangle(int(x_), int(y_), int(x_ + w_), int(y_ + h_))
    shape = face_landmark_predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]
    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)
    leftClosed = int(leftEAR < eyeClosedThreshold)
    rightClosed = int(rightEAR < eyeClosedThreshold)

    blink = 0
    if leftClosed + rightClosed > 0:
        blink = 1

    keyPattern.update(blink)
    similarity = keyPattern.similarity()
    max_similarity = keyPattern.max_similarity()

    for (x, y) in leftEye:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    for (x, y) in rightEye:
        cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)

    memory_str = np.array2string(keyPattern.pattern, precision=1, separator=',', suppress_small=True)

    cv2.putText(frame, "pattern: {:s}".format(memory_str), (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # cv2.putText(frame, "leftEAR: {:.2f}".format(leftEAR), (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.putText(frame, "rightEAR: {:.2f}".format(rightEAR), (260, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    #
    cv2.putText(frame, "leftClosed: {:d}".format(leftClosed), (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.putText(frame, "rightClosed: {:d}".format(rightClosed), (260, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255),
                2)

    # cv2.putText(frame, "sim: {:.2f}".format(similarity), (260, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    # cv2.putText(frame, "maxsim: {:.2f}".format(max_similarity), (240, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255),
    #             2)

    return similarity, max_similarity
