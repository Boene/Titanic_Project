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
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from python_playground.titanic_project.notebooks.functions import eval_modell

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

daten = pd.read_csv("../data/titanic.csv")

#--- Daten clean up ---  

daten["Cabin"] = daten["Cabin"].fillna("unknown")

daten["Age"] = daten["Age"].fillna(daten["Age"].median())

daten["Embarked"] = daten["Embarked"].fillna("unknown")

#----------------------

#--- One Hot Encode ---

ctg_columns = ["Sex","Embarked"]
encoder = OneHotEncoder(sparse_output=False)

ctg_col_encoded = encoder.fit_transform(daten[ctg_columns])

df_encoded = pd.DataFrame(ctg_col_encoded, columns=encoder.get_feature_names_out(ctg_columns))

daten_encoded = pd.concat([daten, df_encoded], axis=1)
daten_encoded = daten_encoded.drop(ctg_columns, axis=1)

#print(daten_encoded)

#----------------------

#--- Data fitting and testing (One Hot) ---

X = daten_encoded[["Pclass", "Sex_female", "Sex_male", "Age", "Fare", "Embarked_C", "Embarked_Q", "Embarked_S", "Embarked_unknown"]]
y = daten_encoded["Survived"]

#----------------------

#--- Cross Validation & Data fitting DecisionTree + RandomForest ---

for est in [5,10,50,100,200]:
    eval_modell(RandomForestClassifier(max_depth=8,random_state=42,n_estimators=est),X,y,c_report=True)
