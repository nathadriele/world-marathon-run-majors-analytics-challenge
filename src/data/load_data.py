import pandas as pd
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config import (
    RAW_DIR,
    PROCESSED_DIR,
    RACE_NAMES,
    YEARS,
    PERFORMANCE_CATEGORIES,
    AGE_GROUPS,
)


def load_csv(filepath: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(filepath)
        return df
    except Exception:
        return pd.DataFrame()


def save_csv(df: pd.DataFrame, filepath: str, index: bool = False) -> bool:
    try:
        directory = os.path.dirname(filepath)
        if directory:
            os.makedirs(directory, exist_ok=True)
        df.to_csv(filepath, index=index)
        return True
    except Exception:
        return False


def load_processed_data(filename: str) -> pd.DataFrame:
    filepath = os.path.join(PROCESSED_DIR, filename)
    return load_csv(filepath)


def load_raw_data(filename: str) -> pd.DataFrame:
    filepath = os.path.join(RAW_DIR, filename)
    return load_csv(filepath)


def save_processed_data(df: pd.DataFrame, filename: str) -> bool:
    filepath = os.path.join(PROCESSED_DIR, filename)
    return save_csv(df, filepath)


def get_data_filepaths() -> dict:
    filepaths = {
        "marathon_results": os.path.join(PROCESSED_DIR, "marathon_results.csv"),
        "winners": os.path.join(PROCESSED_DIR, "winners.csv"),
        "race_metadata": os.path.join(PROCESSED_DIR, "race_metadata.csv"),
        "brazil_analysis": os.path.join(PROCESSED_DIR, "brazil_analysis.csv"),
        "pace_splits": os.path.join(PROCESSED_DIR, "pace_splits.csv"),
        "modeling_dataset": os.path.join(PROCESSED_DIR, "modeling_dataset.csv"),
    }
    return filepaths


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> tuple:
    missing_columns = [col for col in required_columns if col not in df.columns]
    is_valid = len(missing_columns) == 0
    return (is_valid, missing_columns)


def get_race_filter_options(df: pd.DataFrame) -> dict:
    filter_options = {}

    if "year" in df.columns:
        filter_options["years"] = sorted(df["year"].dropna().unique().tolist())
    else:
        filter_options["years"] = YEARS

    if "race" in df.columns:
        filter_options["races"] = sorted(df["race"].dropna().unique().tolist())
    elif "race_name" in df.columns:
        filter_options["races"] = sorted(df["race_name"].dropna().unique().tolist())
    else:
        filter_options["races"] = RACE_NAMES

    if "city" in df.columns:
        filter_options["cities"] = sorted(df["city"].dropna().unique().tolist())
    else:
        filter_options["cities"] = []

    if "country" in df.columns:
        filter_options["countries"] = sorted(df["country"].dropna().unique().tolist())
    else:
        filter_options["countries"] = []

    if "gender" in df.columns:
        filter_options["genders"] = sorted(df["gender"].dropna().unique().tolist())
    elif "sex" in df.columns:
        filter_options["genders"] = sorted(df["sex"].dropna().unique().tolist())
    else:
        filter_options["genders"] = []

    if "age_group" in df.columns:
        filter_options["age_groups"] = sorted(df["age_group"].dropna().unique().tolist())
    else:
        filter_options["age_groups"] = AGE_GROUPS

    if "performance_category" in df.columns:
        filter_options["performance_categories"] = sorted(
            df["performance_category"].dropna().unique().tolist()
        )
    else:
        filter_options["performance_categories"] = PERFORMANCE_CATEGORIES

    if "nationality" in df.columns:
        filter_options["nationalities"] = sorted(
            df["nationality"].dropna().unique().tolist()
        )
    elif "country_of_origin" in df.columns:
        filter_options["nationalities"] = sorted(
            df["country_of_origin"].dropna().unique().tolist()
        )
    else:
        filter_options["nationalities"] = []

    if "division" in df.columns:
        filter_options["divisions"] = sorted(df["division"].dropna().unique().tolist())
    else:
        filter_options["divisions"] = []

    return filter_options
