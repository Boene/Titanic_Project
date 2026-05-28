from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.base import ClassifierMixin

### Split numerical and categorial Features ###

def create_preprocessor(numerical:list, categorical:list, /, num_strategy="median", cat_strategy="constant", cat_fill_value="unknown", enc_handle_unknown="ignore"):
    numerical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy=num_strategy)),
    ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy=cat_strategy,fill_value=cat_fill_value)),
    ("encoder", OneHotEncoder(handle_unknown=enc_handle_unknown)) 
    ])
    
    prepro = ColumnTransformer([
    ("numerical", numerical_pipeline, numerical),
    ("categorical",categorical_pipeline, categorical)
    ])

    return prepro

### Join the Pipelines ###

def create_pipeline (modell, preprocessor):
    if not isinstance(modell, ClassifierMixin):
        raise ValueError("modell must be a classifier")
    
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("modell", modell)
    ])
    return pipe
