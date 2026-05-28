import pandas as pd
import numpy as np
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    mean_absolute_percentage_error,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import cross_val_score, learning_curve
import plotly.graph_objects as go
import plotly.express as px


def regression_metrics(y_true, y_pred) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    return {
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "mape": mape
    }


def classification_metrics(y_true, y_pred, y_prob=None) -> dict:
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    metrics = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
    if y_prob is not None:
        try:
            if y_prob.ndim == 2 and y_prob.shape[1] == 2:
                roc_auc = roc_auc_score(y_true, y_prob[:, 1])
            else:
                roc_auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="weighted")
            metrics["roc_auc"] = roc_auc
        except ValueError:
            pass
    return metrics


def create_model_comparison_chart(results: dict, metric: str) -> dict:
    model_names = []
    metric_values = []
    for model_name, model_results in results.items():
        model_names.append(model_name)
        metric_values.append(model_results.get(metric, 0))
    fig = go.Figure(
        data=[
            go.Bar(
                x=model_names,
                y=metric_values,
                marker_color=np.linspace(0, 1, len(model_names)),
                marker_colorscale="Viridis"
            )
        ]
    )
    fig.update_layout(
        title=f"Model Comparison - {metric}",
        xaxis_title="Model",
        yaxis_title=metric,
        showlegend=False
    )
    return fig.to_dict()


def cross_validate_model(model, X, y, cv: int = 5, scoring: str = "neg_mean_absolute_error") -> dict:
    scores = cross_val_score(model, X, y, cv=cv, scoring=scoring)
    mean_score = np.mean(scores)
    std_score = np.std(scores)
    return {
        "mean_score": mean_score,
        "std_score": std_score,
        "all_scores": scores.tolist()
    }


def plot_confusion_matrix(y_true, y_pred, labels: list) -> go.Figure:
    cm = confusion_matrix(y_true, y_pred)
    fig = go.Figure(
        data=go.Heatmap(
            z=cm,
            x=labels,
            y=labels,
            colorscale="Blues",
            text=cm,
            texttemplate="%{text}",
            textfont={"size": 12},
            showscale=True
        )
    )
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )
    return fig


def plot_regression_results(y_true, y_pred, model_name: str) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=y_true,
            y=y_pred,
            mode="markers",
            name="Predictions",
            marker=dict(color="blue", opacity=0.5, size=5)
        )
    )
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            name="Perfect Prediction",
            line=dict(color="red", dash="dash")
        )
    )
    fig.update_layout(
        title=f"Actual vs Predicted - {model_name}",
        xaxis_title="Actual Values",
        yaxis_title="Predicted Values"
    )
    return fig


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    top_features = importance_df.head(top_n).sort_values("importance", ascending=True)
    fig = go.Figure(
        go.Bar(
            x=top_features["importance"],
            y=top_features["feature"],
            orientation="h",
            marker_color=top_features["importance"],
            marker_colorscale="Viridis"
        )
    )
    fig.update_layout(
        title=f"Top {top_n} Feature Importances",
        xaxis_title="Importance",
        yaxis_title="Feature",
        height=max(400, top_n * 25)
    )
    return fig


def generate_model_report(results: dict) -> pd.DataFrame:
    rows = []
    for model_name, model_results in results.items():
        row = {"model": model_name}
        for metric_name, metric_value in model_results.items():
            if metric_name not in ("model", "predictions"):
                row[metric_name] = metric_value
        rows.append(row)
    report_df = pd.DataFrame(rows)
    return report_df


def plot_learning_curve(model, X, y, cv: int = 5) -> go.Figure:
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y, cv=cv, n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 10),
        shuffle=True,
        random_state=42
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=train_scores_mean,
            mode="lines+markers",
            name="Training Score",
            line=dict(color="blue"),
            error_y=dict(
                type="data",
                array=train_scores_std,
                visible=True
            )
        )
    )
    fig.add_trace(
        go.Scatter(
            x=train_sizes,
            y=test_scores_mean,
            mode="lines+markers",
            name="Cross-Validation Score",
            line=dict(color="orange"),
            error_y=dict(
                type="data",
                array=test_scores_std,
                visible=True
            )
        )
    )
    fig.update_layout(
        title="Learning Curve",
        xaxis_title="Training Examples",
        yaxis_title="Score"
    )
    return fig
