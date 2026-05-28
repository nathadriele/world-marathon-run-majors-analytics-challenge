import pandas as pd
import numpy as np
import math

MARATHON_DISTANCE_KM = 42.195
MARATHON_DISTANCE_MI = 26.2188


def time_to_seconds(time_str: str) -> int:
    if time_str is None or (isinstance(time_str, float) and math.isnan(time_str)):
        return 0
    if isinstance(time_str, str) and time_str.strip() == "":
        return 0
    parts = time_str.split(":")
    if len(parts) == 3:
        hours = int(parts[0])
        minutes = int(parts[1])
        seconds = int(parts[2])
        return hours * 3600 + minutes * 60 + seconds
    if len(parts) == 2:
        minutes = int(parts[0])
        seconds = int(parts[1])
        return minutes * 60 + seconds
    return 0


def seconds_to_time(total_seconds: float) -> str:
    if total_seconds is None or total_seconds <= 0:
        return "00:00:00"
    total_seconds = int(round(total_seconds))
    hours = total_seconds // 3600
    remaining = total_seconds % 3600
    minutes = remaining // 60
    seconds = remaining % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def calculate_pace_per_km(finish_seconds: float, distance_km: float = 42.195) -> float:
    if finish_seconds <= 0 or distance_km <= 0:
        return 0.0
    return (finish_seconds / 60.0) / distance_km


def calculate_pace_per_mile(finish_seconds: float, distance_km: float = 42.195) -> float:
    if finish_seconds <= 0 or distance_km <= 0:
        return 0.0
    distance_mi = distance_km / 1.60934
    return (finish_seconds / 60.0) / distance_mi


def calculate_average_speed_kmh(finish_seconds: float, distance_km: float = 42.195) -> float:
    if finish_seconds <= 0:
        return 0.0
    return distance_km / (finish_seconds / 3600.0)


def calculate_average_speed_ms(finish_seconds: float, distance_km: float = 42.195) -> float:
    if finish_seconds <= 0:
        return 0.0
    distance_m = distance_km * 1000.0
    return distance_m / finish_seconds


def pace_to_time_str(pace_min_per_km: float, distance_km: float = 42.195) -> str:
    if pace_min_per_km <= 0 or distance_km <= 0:
        return "00:00:00"
    total_seconds = pace_min_per_km * 60.0 * distance_km
    return seconds_to_time(total_seconds)


def identify_negative_split(first_half_seconds: float, second_half_seconds: float) -> bool:
    return second_half_seconds < first_half_seconds


def identify_positive_split(first_half_seconds: float, second_half_seconds: float) -> bool:
    return second_half_seconds > first_half_seconds


def calculate_split_difference(first_half_seconds: float, second_half_seconds: float) -> float:
    return second_half_seconds - first_half_seconds


def calculate_pace_variation(pace_list: list) -> float:
    if pace_list is None or len(pace_list) < 2:
        return 0.0
    return float(np.std(pace_list, ddof=1))


def validate_finish_time(finish_seconds: float) -> bool:
    if finish_seconds is None:
        return False
    return 3600 <= finish_seconds <= 32400


def validate_splits_vs_finish(split_seconds_list: list, finish_seconds: float) -> bool:
    if split_seconds_list is None or len(split_seconds_list) == 0:
        return False
    total_splits = sum(split_seconds_list)
    return abs(total_splits - finish_seconds) <= 30


def detect_time_outliers(times_series: pd.Series) -> pd.Series:
    q1 = times_series.quantile(0.25)
    q3 = times_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr
    return (times_series < lower_bound) | (times_series > upper_bound)


def categorize_performance(finish_seconds: float, gender: str) -> str:
    if gender is None:
        return "Recreational"
    gender_upper = gender.strip().upper()
    if gender_upper in ("M", "MALE", "MEN"):
        if finish_seconds < 8400:
            return "Elite"
        if finish_seconds < 10200:
            return "Advanced"
        if finish_seconds < 12600:
            return "Intermediate"
        return "Recreational"
    if gender_upper in ("F", "FEMALE", "WOMEN"):
        if finish_seconds < 9900:
            return "Elite"
        if finish_seconds < 11400:
            return "Advanced"
        if finish_seconds < 13800:
            return "Intermediate"
        return "Recreational"
    return "Recreational"


def create_age_group(age: int) -> str:
    if age is None:
        return "Unknown"
    if 18 <= age <= 24:
        return "18-24"
    if 25 <= age <= 34:
        return "25-34"
    if 35 <= age <= 44:
        return "35-44"
    if 45 <= age <= 54:
        return "45-54"
    if 55 <= age <= 64:
        return "55-64"
    if age >= 65:
        return "65+"
    return "Unknown"


def get_country_code(country_name: str) -> str:
    mapping = {
        "United States": "US",
        "USA": "US",
        "United Kingdom": "GB",
        "UK": "GB",
        "Great Britain": "GB",
        "Canada": "CA",
        "Australia": "AU",
        "Germany": "DE",
        "France": "FR",
        "Japan": "JP",
        "Italy": "IT",
        "Spain": "ES",
        "Brazil": "BR",
        "Mexico": "MX",
        "Netherlands": "NL",
        "Belgium": "BE",
        "Switzerland": "CH",
        "Austria": "AT",
        "Sweden": "SE",
        "Norway": "NO",
        "Denmark": "DK",
        "Finland": "FI",
        "Ireland": "IE",
        "Portugal": "PT",
        "Poland": "PL",
        "South Africa": "ZA",
        "Kenya": "KE",
        "Ethiopia": "ET",
        "China": "CN",
        "India": "IN",
        "South Korea": "KR",
        "Russia": "RU",
        "Argentina": "AR",
        "Colombia": "CO",
        "Chile": "CL",
        "Peru": "PE",
        "New Zealand": "NZ",
        "Israel": "IL",
        "Turkey": "TR",
        "Singapore": "SG",
        "Malaysia": "MY",
        "Thailand": "TH",
        "Philippines": "PH",
        "Indonesia": "ID",
        "Czech Republic": "CZ",
        "Czechia": "CZ",
        "Romania": "RO",
        "Hungary": "HU",
        "Greece": "GR",
        "Croatia": "HR",
        "Luxembourg": "LU",
        "Iceland": "IS",
    }
    if country_name is None:
        return ""
    return mapping.get(country_name.strip(), "")


def format_pace_str(pace_per_km: float) -> str:
    if pace_per_km is None or pace_per_km < 0:
        return "0:00"
    minutes = int(pace_per_km)
    seconds = round((pace_per_km - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"
