import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

ENCODING_PATH = "../../encodings.pickle"
dataset = pickle.loads(open(ENCODING_PATH, "rb").read())

X = np.array(dataset['encodings'], ndmin=2)
names = np.array(dataset['names'], ndmin=1)
le = LabelEncoder()
y = le.fit_transform(names)
dataset['idx'] = y

cls = MLPClassifier(hidden_layer_sizes=(100, 10), max_iter=10000)
cls.fit(X, y)

with open('../../models/mlp.pickle', 'wb') as f:
    pickle.dump(cls, f)
