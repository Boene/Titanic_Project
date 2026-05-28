import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, FixedThresholdClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

daten = pd.read_csv("../data/titanic.csv")

#--- Daten clean up ---  

daten["Cabin"] = daten["Cabin"].fillna("unknown")

daten["Age"] = daten["Age"].fillna(daten["Age"].median())

daten["Embarked"] = daten["Embarked"].fillna("unknown")

daten_old = daten.copy()

def sex_mapper(sex_str):
    sex_map = {"male": 0, "female": 1}
    return sex_map.get(sex_str)

def sex_str_to_int(data):
    data["Sex"] = data["Sex"].apply(sex_mapper)
    return data

def emb_mapper(emb_str):
    emb_map = {"S": 0, "C" : 1, "Q" : 2, "unknown" : 3}
    return emb_map.get(emb_str)


def emb_str_to_int(data):
    data["Embarked"] = data["Embarked"].fillna("unknown")
    data["Embarked"] = data["Embarked"].apply(emb_mapper)
    return data

daten_old = sex_str_to_int(daten_old)
daten_old = emb_str_to_int(daten_old)

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
"""
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33, random_state=42)

modell = LogisticRegression()
modell.fit(X_train, y_train)
"""
#----------------------

#--- Data fitting and testing (old) ---
"""
X_old = daten_old[["Pclass", "Sex", "Age", "Fare"]]
y_old = daten_old["Survived"]

X_old_train, X_old_test, y_old_train, y_old_test = train_test_split(X_old,y_old,test_size=0.33, random_state=42)

modell_old = LogisticRegression()
modell_old.fit(X_old_train, y_old_train)
"""
#----------------------

#--- Data fitting DecisionTree ---

X_dt_train, X_dt_test, y_dt_train, y_dt_test = train_test_split(X,y,test_size=0.33, random_state=42)

for depth in [2,3,4,5,6,7,8,9,10]:
    modell_dt = DecisionTreeClassifier(max_depth=depth, random_state=42)
    modell_dt.fit(X_dt_train, y_dt_train)
    train_accuracy = modell_dt.score(X_dt_train, y_dt_train)
    test_accuracy = modell_dt.score(X_dt_test, y_dt_test)
    print(f"Depth = {depth}:\n")
    #print(metrics.classification_report(y_dt_test, modell_dt.predict(X_dt_test)))
    print(f"Train Accuracy: {train_accuracy:.3f}")
    print(f"Test Accuracy:  {test_accuracy:.3f}")
    print(modell_dt.feature_importances_)
    print("-" * 30)



"""
print("One Hot:\n")
print(metrics.classification_report(y_test, modell.predict(X_test)))
print("Old:\n")
print(metrics.classification_report(y_old_test, modell_old.predict(X_old_test)))
print("DT:\n")
print(metrics.classification_report(y_dt_test, modell_dt.predict(X_dt_test)))
"""
#plt.figure(figsize=(20,10))
#plot_tree(modell_dt, feature_names=X.columns, filled=True)
#plt.show()