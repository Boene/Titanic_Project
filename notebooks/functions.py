import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, FixedThresholdClassifier, cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import ClassifierMixin

def eval_modell(modell, X, y, /, c_report=False, cv=5, test_size=0.25):
    """
    Parameters
    ----------
    modell: sk learn Estimator
    X: features of data that is being tested
    y: target of data that is being tested
    c_report: True, False (Default)
    cv: int (Default = 5)
    test_size: float from 0 to 1 (Default = 0.25)
    """

    if not isinstance(modell, ClassifierMixin):
        raise ValueError("modell must be a classifier")

    rs = getattr(modell, "random_state", None)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=test_size, random_state=rs)

    print("*"*15)
    print(f"modell: {modell.__class__.__name__} | test_size={test_size} | random_state:{rs}\n")

    scores = cross_val_score(modell, X, y, cv=cv)
    modell.fit(X_train, y_train)
    train_accuracy = modell.score(X_train, y_train)
    test_accuracy = modell.score(X_test, y_test)

    if c_report:
        print("-"*10+"\n")
        print(metrics.classification_report(y_test, modell.predict(X_test)))
        print("-"*10+"\n")

    #print("-"*10 + "\n")
    print(f"Cross-validation scores: {scores}\n")
    print(f"Mean cross-validation score: {scores.mean():.2f} | 1 Sigma: {scores.std()}\n")
    print(f"Train Acc: {train_accuracy} | Test Acc: {test_accuracy}\n" + "*"*15 + "\n")

    return