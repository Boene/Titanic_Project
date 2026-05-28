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

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25, 
    random_state=42
)

### Grid Search ###

param_grid = {
    "modell__max_depth": [5, 8, 11, 15],
    "modell__n_estimators": [30, 100, 300, 500]
}

#param_grid = {
#    "modell__max_depth": [2, 5],
#    "modell__n_estimators": [10, 30]
#}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring="accuracy"
)

grid_search.fit(X_train, y_train)

### Analyze Pipeline ###

transformed_feature_names = (                    # Features got changed by OneHot.
    grid_search                         
    .best_estimator_                             # This addresses the optimal RF. 
    .named_steps["preprocessor"]                 # This addresses the preprocessor, which delivers them to the pipeline.
    .get_feature_names_out()            
)

feature_importances = (
    grid_search
    .best_estimator_
    .named_steps["modell"]                       # The importances are part of the optimal RF that was fitted.
    .feature_importances_
)

importance_df = pd.DataFrame({                   # Combine new features names with their importances.
    "feature": transformed_feature_names,
    "importance": feature_importances
})

importance_df = importance_df.sort_values(       # Sort the importances by value, since we re most interested in their hierarchy.
    by="importance",
    ascending=False
)

#print(importance_df)

### Visualizing Results ###

plt.figure(figsize=(10,6))

sns.barplot(
    x=importance_df["importance"],
    y=importance_df["feature"]
)

plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Feature Importances [RF]")
plt.tight_layout()

plt.show()

"""
results = pd.DataFrame(grid_search.cv_results_)

pivoted_data = results.pivot(
    index="param_modell__max_depth", 
    columns="param_modell__n_estimators", 
    values="mean_test_score"
)

sns.heatmap(
    pivoted_data, 
    annot=True, 
    fmt=".3f",
    cmap="coolwarm"
)

plt.xlabel("n_estimators")
plt.ylabel("max_depth")
plt.title("Grid Search Accuracy")

plt.show()

#print(grid_search.best_score_)
#print(grid_search.best_params_)
#print(grid_search.cv_results_)

"""