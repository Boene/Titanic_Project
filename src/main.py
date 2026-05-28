import os
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.model_selection import train_test_split

import preprocessing, training, evaluate

path = os.path.dirname(os.path.abspath(__file__))
os.chdir(path)

### Load Data ###

daten = pd.read_csv("../data/titanic.csv")

### Define Features, Target, numerical, categorical ###

X = daten[["Pclass", "Sex", "Age", "Fare", "Embarked"]]

y = daten["Survived"]

numerical_cats = ["Pclass", "Age", "Fare"]

categorical_cats = ["Sex", "Embarked"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25, 
    random_state=42
)

### Set Preprocessor & Pipeline ###

preprocessor = preprocessing.create_preprocessor(numerical_cats, categorical_cats)

pipe = preprocessing.create_pipeline(RandomForestClassifier(), preprocessor)

### Configure and run GridSearch ###

param_grid = {
    "modell__max_depth": [5, 15],
    "modell__n_estimators": [30, 100]    
}

grid_search = training.run_grid_search(X_train, y_train, pipe, param_grid, cv=5, scoring="accuracy")

importance_df, test_accuracy = evaluate.analyze_grid(grid_search, X_test, y_test)

evaluate.show_gridsearch_analysis(importance_df, test_accuracy)

### Save the best Model ###

optimal_model = grid_search.best_estimator_

joblib.dump(optimal_model,"../models/my_model.pkl")
