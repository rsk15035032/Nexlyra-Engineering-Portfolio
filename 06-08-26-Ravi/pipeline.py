from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder



# Feature Lists

null_value_column = [
    "Square_Footage",
    "Crime_Index"
]

continuous_features = [
    "Lot_Size_Acres",
    "Distance_To_City_Center_KM",
    "Transit_to_Distance_Ratio",
    "Transit_Score",
    "Square_Footage",
    "Property_Age",
    "Environmental_Risk_Score",
    "Crime_Index"
]

onehot_features = [
    "Property_Type",
    "Zoning_Code",
    "Country_Code"
]

ordinal_features = [
    "Energy_Efficiency_Rating"
]

ordinal_categories = [
    ["F", "E", "D", "C", "B", "A"]
]


# Numeric Pipeline
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# One-Hot Pipeline
onehot_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


# Ordinal Pipeline
ordinal_pipeline = Pipeline([
    (
        "encoder",
        OrdinalEncoder(
            categories=ordinal_categories,
            handle_unknown="use_encoded_value",
            unknown_value=-1
        )
    )
])


# Combine Everything
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, continuous_features),
        ("cat", onehot_pipeline, onehot_features),
        ("ord", ordinal_pipeline, ordinal_features),
    ],
    remainder="drop"
)


# Fit on Training Data
## X_train_processed = preprocessor.fit_transform(X_train)

# Transform Test Data
## X_test_processed = preprocessor.transform(X_test)

