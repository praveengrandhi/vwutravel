import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import os
import mlflow

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("tourism-package-experiment")

Xtrain = pd.read_csv("Xtrain.csv")
Xtest = pd.read_csv("Xtest.csv")
ytrain = pd.read_csv("ytrain.csv").values.ravel()
ytest = pd.read_csv("ytest.csv").values.ravel()

numeric_features = [
    'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 
    'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
    'Passport', 'PitchSatisfactionScore', 'OwnCar', 
    'NumberOfChildrenVisiting', 'MonthlyIncome'
]

categorical_features = [
    'TypeofContact', 'Occupation', 'Gender', 'ProductPitched', 
    'MaritalStatus', 'Designation'
]

preprocessor = make_column_transformer(
    (StandardScaler(), numeric_features),
    (OneHotEncoder(handle_unknown='ignore'), categorical_features)
)

xgb_model = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='logloss')
model_pipeline = make_pipeline(preprocessor, xgb_model)

param_grid = {
    'xgbclassifier__n_estimators': [50, 100],
    'xgbclassifier__max_depth': [3, 5],
    'xgbclassifier__learning_rate': [0.05, 0.1],
    'xgbclassifier__subsample': [0.8, 1.0]
}

with mlflow.start_run() as parent_run:
    grid_search = GridSearchCV(model_pipeline, param_grid, cv=3, n_jobs=-1, scoring='f1')
    grid_search.fit(Xtrain, ytrain)

    cv_results = grid_search.cv_results_
    for i in range(len(cv_results['params'])):
        params = cv_results['params'][i]
        f1_mean = cv_results['mean_test_score'][i]
        with mlflow.start_run(nested=True):
            mlflow.log_params(params)
            mlflow.log_metric("mean_cv_f1", f1_mean)

    mlflow.log_params(grid_search.best_params_)
    best_model = grid_search.best_estimator_

    ytrain_pred = best_model.predict(Xtrain)
    ytest_pred = best_model.predict(Xtest)
    ytest_proba = best_model.predict_proba(Xtest)[:, 1]

    metrics = {
        "train_accuracy": accuracy_score(ytrain, ytrain_pred),
        "test_accuracy": accuracy_score(ytest, ytest_pred),
        "train_precision": precision_score(ytrain, ytrain_pred),
        "test_precision": precision_score(ytest, ytest_pred),
        "train_recall": recall_score(ytrain, ytrain_pred),
        "test_recall": recall_score(ytest, ytest_pred),
        "train_f1": f1_score(ytrain, ytrain_pred),
        "test_f1": f1_score(ytest, ytest_pred),
        "test_roc_auc": roc_auc_score(ytest, ytest_proba)
    }

    mlflow.log_metrics(metrics)

    os.makedirs("tourism_project/deployment", exist_ok=True)
    model_path = "tourism_project/deployment/best_tourism_model.joblib"
    joblib.dump(best_model, model_path)
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved to {model_path}")

