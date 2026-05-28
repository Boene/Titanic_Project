import joblib
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression

from sklearn.model_selection import train_test_split

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

### Daten laden ###

daten = pd.read_csv("../data/titanic.csv")

### Gespeichertes Modell laden ###

loaded_model = joblib.load("../models/my_model.pkl")

### Testdaten und Target bereitstellen ###

X = daten[["Pclass", "Sex", "Age", "Fare", "Embarked"]]

y = daten["Survived"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25, 
    random_state=42
)

### Vergleich von Vorhersage und Originaltarget ###

loaded_predict = loaded_model.predict(X_test)

print(f"Prediction of loaded Modell on y_test: {loaded_predict}")
