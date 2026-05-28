import pandas as pd
import numpy as np


def validate_marathon_data(df: pd.DataFrame) -> dict:
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
        "row_count": len(df),
    }

    required_columns = [
        "runner_name",
        "race_name",
        "year",
        "finish_time",
        "gender",
        "age",
    ]

    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        report["errors"].append(f"Missing required columns: {missing_cols}")
        report["is_valid"] = False

    if "finish_time" in df.columns:
        expected_dtype = [np.float64, np.int64, np.float32, np.int32, float, int]
        if df["finish_time"].dtype not in expected_dtype:
            try:
                numeric_check = pd.to_numeric(df["finish_time"], errors="coerce")
                if numeric_check.isna().all():
                    report["errors"].append("finish_time column cannot be converted to numeric")
                    report["is_valid"] = False
            except Exception:
                report["errors"].append("finish_time column has invalid data type")
                report["is_valid"] = False

    critical_columns = ["runner_name", "race_name", "year", "finish_time"]
    for col in critical_columns:
        if col in df.columns:
            null_count = df[col].isna().sum()
            if null_count > 0:
                report["errors"].append(f"Column '{col}' has {null_count} null values")
                report["is_valid"] = False

    if "finish_time" in df.columns:
        finish_times = pd.to_numeric(df["finish_time"], errors="coerce")
        below_min = finish_times[finish_times < 3600]
        above_max = finish_times[finish_times > 32400]
        if len(below_min) > 0:
            report["warnings"].append(
                f"{len(below_min)} rows with finish time below 1 hour"
            )
        if len(above_max) > 0:
            report["warnings"].append(
                f"{len(above_max)} rows with finish time above 9 hours"
            )

    if "age" in df.columns:
        ages = pd.to_numeric(df["age"], errors="coerce")
        below_min_age = ages[ages < 10]
        above_max_age = ages[ages > 100]
        if len(below_min_age) > 0:
            report["warnings"].append(
                f"{len(below_min_age)} rows with age below 10"
            )
        if len(above_max_age) > 0:
            report["warnings"].append(
                f"{len(above_max_age)} rows with age above 100"
            )

    return report


def validate_split_consistency(df: pd.DataFrame) -> dict:
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
    }

    split_columns = [col for col in df.columns if "split" in col.lower() or "km" in col.lower() or "half" in col.lower() or "5k" in col.lower() or "10k" in col.lower() or "15k" in col.lower() or "20k" in col.lower() or "25k" in col.lower() or "30k" in col.lower() or "35k" in col.lower() or "40k" in col.lower()]

    if not split_columns:
        report["warnings"].append("No split columns found in DataFrame")
        return report

    split_columns_sorted = sorted(split_columns)

    for idx, row in df.iterrows():
        split_values = []
        for col in split_columns_sorted:
            val = row.get(col)
            if pd.notna(val):
                try:
                    split_values.append(float(val))
                except (ValueError, TypeError):
                    continue

        for i in range(len(split_values) - 1):
            if split_values[i + 1] < split_values[i]:
                report["errors"].append(
                    f"Row {idx}: splits are not monotonically increasing"
                )
                report["is_valid"] = False
                break

        for val in split_values:
            if val < 0:
                report["errors"].append(f"Row {idx}: negative split value found")
                report["is_valid"] = False
                break

        if "finish_time" in df.columns and len(split_values) > 0:
            finish = row.get("finish_time")
            if pd.notna(finish):
                try:
                    finish_val = float(finish)
                    if split_values[-1] > finish_val:
                        report["errors"].append(
                            f"Row {idx}: last split ({split_values[-1]}) exceeds finish time ({finish_val})"
                        )
                        report["is_valid"] = False
                except (ValueError, TypeError):
                    pass

    return report


def validate_winners_data(df: pd.DataFrame) -> dict:
    report = {
        "is_valid": True,
        "errors": [],
        "warnings": [],
    }

    major_races = [
        "Boston",
        "London",
        "Berlin",
        "Chicago",
        "New York City",
        "Tokyo",
    ]

    if "race_name" not in df.columns or "year" not in df.columns:
        report["errors"].append("Missing race_name or year columns")
        report["is_valid"] = False
        return report

    race_names_lower = [r.lower() for r in df["race_name"].dropna().unique()]
    for race in major_races:
        if race.lower() not in race_names_lower:
            report["warnings"].append(f"Major race not found: {race}")

    if "year" in df.columns:
        years = df["year"].unique()
        for year in years:
            year_data = df[df["year"] == year]
            races_in_year = year_data["race_name"].nunique()
            if races_in_year < 6:
                report["warnings"].append(
                    f"Year {year}: only {races_in_year} races represented (expected 6)"
                )

    if "runner_name" in df.columns and "year" in df.columns and "race_name" in df.columns:
        duplicates = df[df.duplicated(subset=["runner_name", "year", "race_name"], keep=False)]
        if len(duplicates) > 0:
            report["errors"].append(
                f"{len(duplicates)} duplicate winner entries found"
            )
            report["is_valid"] = False

    if "time" in df.columns:
        invalid_times = []
        for idx, val in df["time"].items():
            if pd.notna(val):
                val_str = str(val)
                if ":" not in val_str:
                    invalid_times.append(idx)
                else:
                    parts = val_str.split(":")
                    if len(parts) < 2 or len(parts) > 3:
                        invalid_times.append(idx)
                    else:
                        try:
                            [float(p) for p in parts]
                        except ValueError:
                            invalid_times.append(idx)

        if invalid_times:
            report["errors"].append(
                f"{len(invalid_times)} rows with invalid time format"
            )
            report["is_valid"] = False

    return report


def generate_data_report(df: pd.DataFrame) -> pd.DataFrame:
    report_data = []

    for col in df.columns:
        null_count = df[col].isna().sum()
        null_pct = (null_count / len(df)) * 100 if len(df) > 0 else 0
        unique_count = df[col].nunique()
        sample_values = df[col].dropna().head(3).tolist()

        report_data.append(
            {
                "column_name": col,
                "dtype": str(df[col].dtype),
                "null_count": null_count,
                "null_pct": round(null_pct, 2),
                "unique_values": unique_count,
                "sample_values": sample_values,
            }
        )

    report_df = pd.DataFrame(report_data)

    return report_df


def check_data_freshness(df: pd.DataFrame, date_column: str = "year", max_age_years: int = 2) -> dict:
    report = {
        "is_fresh": True,
        "latest_year": None,
        "max_age_years": max_age_years,
        "message": "",
    }

    if date_column not in df.columns:
        report["is_fresh"] = False
        report["message"] = f"Column '{date_column}' not found in DataFrame"
        return report

    current_year = pd.Timestamp.now().year

    years = pd.to_numeric(df[date_column], errors="coerce").dropna()
    if len(years) == 0:
        report["is_fresh"] = False
        report["message"] = "No valid year values found"
        return report

    latest_year = int(years.max())
    report["latest_year"] = latest_year

    age = current_year - latest_year
    if age > max_age_years:
        report["is_fresh"] = False
        report["message"] = (
            f"Data is {age} years old (max allowed: {max_age_years}). "
            f"Latest year: {latest_year}, current year: {current_year}"
        )
    else:
        report["message"] = (
            f"Data is fresh. Latest year: {latest_year}, current year: {current_year}"
        )

    return report
