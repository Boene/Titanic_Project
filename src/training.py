from sklearn.model_selection import train_test_split, GridSearchCV

def run_grid_search(features, target, pipeline, param_grid, /, cv=5, scoring="accuracy"):

    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring
    )

    grid.fit(features, target)

    return grid

