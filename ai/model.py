# ai/model.py
import numpy as np
from sklearn.linear_model import SGDClassifier

CLASS_NAMES = np.array(['Hold','Buy','Sell'])  # fixed order

def make_model(random_state: int = 42) -> SGDClassifier:
    # logistic regression with partial_fit for online learning
    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=1e-4,
                        max_iter=1, learning_rate='optimal',
                        random_state=random_state, warm_start=True)
    return clf

def initial_partial_fit(clf: SGDClassifier, X, y):
    classes = np.arange(len(CLASS_NAMES))
    clf.partial_fit(X, y, classes=classes)
    return clf
