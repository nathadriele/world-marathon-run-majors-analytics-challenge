import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def predict_finish_time(model, input_data: dict) -> dict:
    input_df = pd.DataFrame([input_data])
    feature_columns = [
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
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    predicted_seconds = float(model.predict(input_df)[0])
    hours = int(predicted_seconds // 3600)
    minutes = int((predicted_seconds % 3600) // 60)
    seconds = int(predicted_seconds % 60)
    predicted_finish_time = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    marathon_km = 42.195
    marathon_mile = 26.2188
    pace_per_km_seconds = predicted_seconds / marathon_km
    pace_km_min = int(pace_per_km_seconds // 60)
    pace_km_sec = int(pace_per_km_seconds % 60)
    predicted_pace_per_km = f"{pace_km_min}:{pace_km_sec:02d} /km"
    pace_per_mile_seconds = predicted_seconds / marathon_mile
    pace_mile_min = int(pace_per_mile_seconds // 60)
    pace_mile_sec = int(pace_per_mile_seconds % 60)
    predicted_pace_per_mile = f"{pace_mile_min}:{pace_mile_sec:02d} /mile"
    return {
        "predicted_finish_seconds": predicted_seconds,
        "predicted_finish_time": predicted_finish_time,
        "predicted_pace_per_km": predicted_pace_per_km,
        "predicted_pace_per_mile": predicted_pace_per_mile
    }


def predict_performance_category(model, input_data: dict, label_encoder) -> str:
    input_df = pd.DataFrame([input_data])
    feature_columns = [
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
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    predicted_class = model.predict(input_df)[0]
    predicted_category = label_encoder.inverse_transform([predicted_class])[0]
    return predicted_category


def batch_predict(model, df: pd.DataFrame) -> np.ndarray:
    feature_columns = [
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
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    predictions = model.predict(df[feature_columns])
    return predictions


def predict_with_confidence(model, input_data: dict) -> dict:
    input_df = pd.DataFrame([input_data])
    feature_columns = [
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
    for col in feature_columns:
        if col not in input_df.columns:
            input_df[col] = 0
    input_df = input_df[feature_columns]
    main_prediction = float(model.predict(input_df)[0])
    if hasattr(model, "estimators_"):
        tree_predictions = []
        for estimator in model.estimators_:
            if hasattr(estimator, "predict"):
                tree_predictions.append(float(estimator.predict(input_df)[0]))
        tree_predictions = np.array(tree_predictions)
        lower_bound = float(np.percentile(tree_predictions, 2.5))
        upper_bound = float(np.percentile(tree_predictions, 97.5))
        std_dev = float(np.std(tree_predictions))
        mean_prediction = float(np.mean(tree_predictions))
    else:
        lower_bound = main_prediction
        upper_bound = main_prediction
        std_dev = 0.0
        mean_prediction = main_prediction
    return {
        "prediction": main_prediction,
        "lower_bound_95": lower_bound,
        "upper_bound_95": upper_bound,
        "std_deviation": std_dev,
        "mean_tree_prediction": mean_prediction
    }


def create_prediction_profile(runner_data: dict) -> pd.DataFrame:
    display_names = {
        "year": "Year",
        "gender_encoded": "Gender (Encoded)",
        "age": "Age",
        "race_encoded": "Race (Encoded)",
        "runner_country_encoded": "Country (Encoded)",
        "first_half_seconds": "First Half (seconds)",
        "negative_split_flag": "Negative Split",
        "positive_split_flag": "Positive Split",
        "pace_variation": "Pace Variation"
    }
    rows = []
    for key, value in runner_data.items():
        display_key = display_names.get(key, key)
        rows.append({"Parameter": display_key, "Value": value})
    profile_df = pd.DataFrame(rows)
    return profile_df
