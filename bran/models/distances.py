from collections import defaultdict

from collections import defaultdict
import face_recognition
import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin

UNKNOWN = -1


class DistanceMinAvg(BaseEstimator, ClassifierMixin):

    def __init__(self, face2vec=None, confidence=0.5):
        self.face2vec = face2vec
        self.confidence = confidence

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return np.array([self.face_method(encoding) for encoding in X])

    def face_method(self, encoding):
        distances = face_recognition.face_distance(self.face2vec["encodings"], encoding)

        final_idx = UNKNOWN
        counter = defaultdict(lambda: 0)
        summer = defaultdict(lambda: 0)

        for k, v in zip(self.face2vec['idx'], distances):
            counter[k] += 1
            summer[k] += v

        idx, min_avg = min(((idx, summer[idx] / counter[idx]) for idx in self.face2vec['idx']),
                           key=lambda x: x[1])

        if min_avg < (1 - self.confidence):
            final_idx = idx

        return final_idx


class DistanceVoting(BaseEstimator, ClassifierMixin):

    def __init__(self, face2vec=None, confidence=0.5):
        """
        Called when initializing the classifier
        """
        self.face2vec = face2vec
        self.confidence = confidence

    def fit(self, X, y=None):
        return self

    def predict(self, X, y=None):
        return np.array([self.face_compare(encoding) for encoding in X])

    def face_compare(self, X):
        matches = face_recognition.compare_faces(self.face2vec["encodings"], X, tolerance=1 - self.confidence)
        name = UNKNOWN
        # check to see if we have found a match
        if True in matches:
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face was matched
            matched_indexes = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            # loop over the matched indexes and maintain a count for
            # each recognized face face
            for i in matched_indexes:
                name = self.face2vec["idx"][i]
                counts[name] = counts.get(name, 0) + 1

            # determine the recognized face with the largest number
            # of votes (note: in the event of an unlikely tie Python
            # will select first entry in the dictionary)
            name = max(counts, key=counts.get)
        return name
