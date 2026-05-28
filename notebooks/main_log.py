import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, FixedThresholdClassifier
from sklearn import metrics

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

daten = pd.read_csv("../data/titanic.csv")

# -- 1 -- Sex String to Int (0 = Male, 1 = Female)

def sex_mapper(sex_str):
    sex_map = {"male": 0, "female": 1}
    return sex_map.get(sex_str)

def sex_str_to_int(data):
    data["Sex"] = data["Sex"].apply(sex_mapper)
    return data



# --- 2 --- Replace NaN Cabin with "unknown"

daten["Cabin"] = daten["Cabin"].fillna("unknown")
#print(sex_str_to_int(daten))


# --- 3 --- Missing Age Values

daten["Age"] = daten["Age"].fillna(daten["Age"].median())

# --- 4 --- Embarked String to Int (S (Southampton) = 0, C (Chergourg) = 1, Q (Queenstown) = 2. unknown = 3)

def emb_mapper(emb_str):
    emb_map = {"S": 0, "C" : 1, "Q" : 2, "unknown" : 3}
    return emb_map.get(emb_str)


def emb_str_to_int(data):
    data["Embarked"] = data["Embarked"].fillna("unknown")
    data["Embarked"] = data["Embarked"].apply(emb_mapper)
    return data

daten = sex_str_to_int(daten)
daten = emb_str_to_int(daten)

X = daten[["Pclass", "Sex", "Age", "Fare"]]
y = daten["Survived"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)

modell = LogisticRegression()
modell.fit(X_train, y_train)
print(metrics.classification_report(y_test, modell.predict(X_test)))

#print(daten.info())
#print(daten[daten["Embarked"] == 3])
#print(daten.head())
#print("-----")


#Meine Vermutung ist, daß ein voller Datensatz, ohne fehlende Werte, 891 non-Null Werte liefern sollte. Demnach würden bei Age und Cabin einige Einträge fehlen. 
#Daß das Alter als Float angegeben wird, finde ich interessant und ich würde sagen, daß das sehr unüblich ist. 
#min und max bei PassengerID zeigen, daß 891 Passagiere bzw. zumindest IDs für solche vergeben wurden. 
#Der mean Wert von 0,38 bei Survived lässt mich vermuten, daß ca. 38% der Leute aus der Liste überlebt haben. Das Durchschnittsalter lag anscheinend bei 29,7 Jahren und der durchschnittliche Ticketpreis bei 32,2 Dollar (?).
#Es scheint 3 verschiedene Klassen zu geben, was vergleichbar mit heutigen Langstreckenflügen scheint. Überraschend finde ich, daß der Durschnitt hier über 2 liegt. Das lässt mich denken, daß die Liste eher wohlhabende Leute beinhaltet.
#Was Embarked, SibSp und Parch bedeuten, weiß ich nicht.

#print("-----")
#print(daten.describe())
#print("-----")
#print(daten[daten["Sex"] == "male"].count())
#print(daten[(daten["Sex"] == "female") & (daten["Survived"] == 1)].count())

#577 Männer und demnach 891 - 577 = 314 Frauen. 
#Es haben 109 Männer und 233 Frauen überlebt. Das Motto "Frauen und Kinder zuerst" scheint also real gewesen zu sein, da das Verhältnis Frauen/Männer vorher und nachher deutlich gestiegen ist.

#Survivors = daten.groupby("Sex")["Survived"].mean()
#print(Survivors)
#Survivors.plot(kind="bar")

#plt.show()