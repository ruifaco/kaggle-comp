
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Carregamento
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
sample_submission = pd.read_csv("sample_submission.csv")

# Pré-processamento

target = "labels"
X = train.drop(columns=[target])
y = train[target]

categorical_cols = ["category_code"]
numerical_cols = [c for c in X.columns if c not in categorical_cols]

# Pipeline do pré-processamento
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols)
    ]
)

# Base models

rf = RandomForestClassifier(random_state=42, class_weight="balanced")
gb = GradientBoostingClassifier(random_state=42)
hgb = HistGradientBoostingClassifier(random_state=42)


estimators = [
    ("rf", rf),
    ("gb", gb),
    ("hgb", hgb)
]

stack = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
    n_jobs=-1
)

# Grid Search para RF

param_grid = {
    "rf__n_estimators": [200, 500],
    "rf__max_depth": [8, 12, None],
    "rf__min_samples_split": [2, 5, 10]
}

pipe = Pipeline(steps=[("preprocessor", preprocessor), ("model", stack)])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

grid_search = GridSearchCV(pipe, param_grid, cv=cv, scoring="accuracy", n_jobs=-1, verbose=2)
grid_search.fit(X, y)

print("Best params:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)

# Treino final

final_model = grid_search.best_estimator_
final_model.fit(X, y)

# predição

preds = final_model.predict(test)
submission = sample_submission.copy()
submission["labels"] = preds

submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")