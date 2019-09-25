import pickle
import numpy as np
from sklearn import model_selection
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

from bran.cross_validation.plot import plot_confusion_matrix
from bran.models.distances import DistanceMinAvg
from sklearn.model_selection import cross_val_predict
import matplotlib.pyplot as plt


ENCODING_PATH = "../../encodings.pickle"

dataset = pickle.loads(open(ENCODING_PATH, "rb").read())

cv = model_selection.LeaveOneOut()
# cv = None

X = np.array(dataset['encodings'], ndmin=2)
names = np.array(dataset['names'], ndmin=1)
le = LabelEncoder()
y = le.fit_transform(names)
dataset['idx'] = y

classes = np.array([c for c in le.classes_] + ['Unknown'])
# cls = MLPClassifier()
cls = DistanceMinAvg(dataset, 0.7)

y_pred = cross_val_predict(cls, X, y, cv=cv)

cls.fit(X, y)
# print(cls.predict_proba(X))

plot_confusion_matrix(y, y_pred, classes=classes,
                      title='Confusion matrix, without normalization')

# Plot normalized confusion matrix
plot_confusion_matrix(y, y_pred, classes=classes, normalize=True,
                      title='Normalized confusion matrix')

plt.show()

if __name__ == '__main__':
    pass
