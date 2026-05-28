import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib


def prepare_regression_data(df: pd.DataFrame) -> tuple:
    features = [
        "year",
        "gender_encoded",
        "age",
        "race_encoded",
        "runner_country_encoded",
        "first_half_seconds",
        "negative_split_flag",
        "positive_split_flag",
        "pace_variation"
    ]
    X = df[features].copy()
    y = df["finish_seconds"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def prepare_classification_data(df: pd.DataFrame) -> tuple:
    features = [
        "year",
        "gender_encoded",
        "age",
        "race_encoded",
        "runner_country_encoded",
        "first_half_seconds",
        "negative_split_flag",
        "positive_split_flag",
        "pace_variation"
    ]
    X = df[features].copy()
    y = df["performance_category_encoded"].copy()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_regression_models(X_train, y_train, X_test, y_test) -> dict:
    models = {
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.1),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=100, random_state=42),
        "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=100, random_state=42)
    }
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        mape = mean_absolute_percentage_error(y_test, preds)
        results[model_name] = {
            "model": model,
            "predictions": preds,
            "mae": mae,
            "rmse": rmse,
            "r2": r2,
            "mape": mape
        }
    return results


def train_classification_models(X_train, y_train, X_test, y_test) -> dict:
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100, random_state=42),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    results = {}
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average="weighted", zero_division=0)
        recall = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)
        entry = {
            "model": model,
            "predictions": preds,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1
        }
        try:
            y_prob = model.predict_proba(X_test)
            if y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_test, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            entry["roc_auc"] = roc_auc
        except (AttributeError, ValueError):
            pass
        results[model_name] = entry
    return results


def train_clustering(df: pd.DataFrame, n_clusters: int = 4) -> tuple:
    cluster_features = ["finish_seconds", "pace_per_km", "age", "average_speed_kmh"]
    data = df[cluster_features].copy()
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = model.fit_predict(scaled_data)
    return model, labels, scaled_data


def get_feature_importance(model, feature_names: list) -> pd.DataFrame:
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "coef_"):
        importances = np.abs(model.coef_)
        if importances.ndim > 1:
            importances = importances.mean(axis=0)
    else:
        importances = np.zeros(len(feature_names))
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": importances
    })
    importance_df = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    return importance_df


def save_model(model, filepath: str) -> bool:
    try:
        joblib.dump(model, filepath)
        return True
    except Exception:
        return False


def load_model(filepath: str):
    return joblib.load(filepath)
