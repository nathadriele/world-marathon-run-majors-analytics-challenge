import pandas as pd
import numpy as np


def clean_marathon_results(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df = df.drop_duplicates(subset=["runner_name", "race_name", "year"], keep="first")

    string_cols = df.select_dtypes(include=["object"]).columns
    for col in string_cols:
        df[col] = df[col].str.strip()

    if "gender" in df.columns:
        gender_map = {"M": "Male", "F": "Female", "m": "Male", "f": "Female"}
        df["gender"] = df["gender"].replace(gender_map)

    if "status" in df.columns:
        status_mask = df["status"].str.upper().isin(["DNF", "DNS"])
        df = df[~status_mask].copy()

    if "age" in df.columns:
        median_age = df["age"].median()
        df["age"] = df["age"].fillna(median_age)

    if "finish_time" in df.columns:
        df = df[df["finish_time"] > 0]
        df = df[df["finish_time"] <= 32400]

    df = df.reset_index(drop=True)

    return df


def clean_winners_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "status" in df.columns:
        invalid_status = ["cancelled", "postponed", "Canceled", "Postponed",
                          "CANCELLED", "POSTPONED"]
        df = df[~df["status"].isin(invalid_status)]

    if "country" in df.columns:
        df = standardize_country_names(df, "country")

    if "time" in df.columns:
        df = df[df["time"].notna()]
        valid_time = df["time"].apply(lambda x: isinstance(x, str) and ":" in str(x))
        df = df[valid_time]

    df = df.drop_duplicates()

    df = df.reset_index(drop=True)

    return df


def standardize_country_names(df: pd.DataFrame, column: str) -> pd.DataFrame:
    df = df.copy()

    country_map = {
        "KEN": "Kenya",
        "ETH": "Ethiopia",
        "USA": "United States",
        "US": "United States",
        "U.S.A.": "United States",
        "United States of America": "United States",
        "GBR": "United Kingdom",
        "UK": "United Kingdom",
        "Great Britain": "United Kingdom",
        "GB": "United Kingdom",
        "JPN": "Japan",
        "GER": "Germany",
        "DEU": "Germany",
        "BRA": "Brazil",
        "FRA": "France",
        "AUS": "Australia",
        "ITA": "Italy",
        "ESP": "Spain",
        "CAN": "Canada",
        "RUS": "Russia",
        "CHN": "China",
        "SUI": "Switzerland",
        "SWE": "Sweden",
        "NOR": "Norway",
        "DEN": "Denmark",
        "NED": "Netherlands",
        "HOL": "Netherlands",
        "BEL": "Belgium",
        "POR": "Portugal",
        "POL": "Poland",
        "CZE": "Czech Republic",
        "AUT": "Austria",
        "NZL": "New Zealand",
        "NZ": "New Zealand",
        "IRL": "Ireland",
        "ARG": "Argentina",
        "MEX": "Mexico",
        "KOR": "South Korea",
        "RSA": "South Africa",
        "ZIM": "Zimbabwe",
        "UGA": "Uganda",
        "TAN": "Tanzania",
        "ERI": "Eritrea",
        "MAR": "Morocco",
        "DJI": "Djibouti",
        "BHR": "Bahrain",
        "QAT": "Qatar",
        "TUR": "Turkey",
        "ISR": "Israel",
        "COL": "Colombia",
        "ECU": "Ecuador",
        "PER": "Peru",
        "CHI": "Chile",
        "UAE": "United Arab Emirates",
        "KSA": "Saudi Arabia",
        "CRO": "Croatia",
        "ROM": "Romania",
        "ROU": "Romania",
        "HUN": "Hungary",
        "SRB": "Serbia",
        "SLO": "Slovenia",
        "SVK": "Slovakia",
        "FIN": "Finland",
        "ISL": "Iceland",
        "GRE": "Greece",
        "GRC": "Greece",
    }

    df[column] = df[column].replace(country_map)

    return df


def handle_missing_values(df: pd.DataFrame, strategy: str = "median") -> pd.DataFrame:
    df = df.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    categorical_cols = df.select_dtypes(include=["object"]).columns

    for col in numeric_cols:
        if df[col].isna().any():
            if strategy == "median":
                fill_value = df[col].median()
            elif strategy == "mean":
                fill_value = df[col].mean()
            else:
                fill_value = df[col].median()
            df[col] = df[col].fillna(fill_value)

    for col in categorical_cols:
        if df[col].isna().any():
            mode_values = df[col].mode()
            if len(mode_values) > 0:
                df[col] = df[col].fillna(mode_values[0])

    return df


def remove_outliers(df: pd.DataFrame, column: str, method: str = "iqr", threshold: float = 1.5) -> pd.DataFrame:
    df = df.copy()

    if column not in df.columns:
        return df

    if method == "iqr":
        q1 = df[column].quantile(0.25)
        q3 = df[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    elif method == "zscore":
        col_mean = df[column].mean()
        col_std = df[column].std()
        if col_std == 0:
            return df
        z_scores = np.abs((df[column] - col_mean) / col_std)
        df = df[z_scores <= threshold]

    df = df.reset_index(drop=True)

    return df
