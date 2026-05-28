import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error


### Analyze Pipeline ###

def analyze_grid(grid_search, test_features, test_target):

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

    test_accuracy = grid_search.score(test_features, test_target)

    return importance_df, test_accuracy

def show_gridsearch_analysis(importance_df, test_accuracy):

    print(f"Accuracy on test data: {test_accuracy}")

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
    return