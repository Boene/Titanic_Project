import matplotlib.pyplot as plt
import math
import os
import pandas as pd
import numpy as np
import seaborn as sns
import joblib
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split, FixedThresholdClassifier, cross_val_score, GridSearchCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.svm import SVC
from python_playground.titanic_project.notebooks.functions import eval_modell

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

daten = pd.read_csv("../data/titanic.csv")

### Features and Target ###

X = daten[["Pclass", "Sex", "Age", "Fare", "Embarked"]]

y = daten["Survived"]

### Split numerical and categorial Features ###

numerical_cats = ["Pclass", "Age", "Fare"]

categorical_cats = ["Sex", "Embarked"]

### Preparing Pipelines ###

numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="constant",fill_value="unknown")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

### Join the Pipelines ###

preprocessor = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical_cats),
    ("categorical",categorical_pipeline, categorical_cats)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("modell", RandomForestClassifier())
])

### Split Data ###

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.25, random_state=42)

### Training and Prediction ###

pipeline.fit(X_train, y_train)

prediction = pipeline.predict(X_test)

### View Data ###

encoded_cols = pipeline.named_steps["preprocessor"].named_transformers_["categorical"]["encoder"].get_feature_names_out(categorical_cats)
#print(encoded_cols)

### Cross Validation ###

scores = cross_val_score(pipeline, X, y, cv=5)

### Evaluate Modell ###

mse = mean_squared_error(y_test, prediction)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, prediction)
mae = mean_absolute_error(y_test, prediction)

train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = pipeline.score(X_test, y_test)

### Results ###

print(f"Mean squared: {mse}")
print(f"R2: {r2}")
print(f"Mean absolute: {mae}")
print(f"Train Acc: {train_accuracy} | Test Acc: {test_accuracy}\n" + "*"*15 + "\n")
print(f"Cross-validation scores: {scores}")
