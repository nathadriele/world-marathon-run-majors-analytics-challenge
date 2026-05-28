import pandas as pd
import numpy as np
from src.utils.helpers import (
    time_to_seconds,
    seconds_to_time,
    calculate_pace_per_km,
    calculate_pace_per_mile,
    calculate_average_speed_kmh,
)

MARATHON_DISTANCE_KM = 42.195
HALF_MARATHON_DISTANCE_KM = 21.0975
SPLIT_DISTANCES = {
    "5k": 5,
    "10k": 10,
    "15k": 15,
    "20k": 20,
    "half_marathon": 21.0975,
    "25k": 25,
    "30k": 30,
    "35k": 35,
    "40k": 40,
}
MILE_TO_KM = 1.60934


def calculate_all_running_metrics(finish_seconds: float) -> dict:
    finish_time_str = seconds_to_time(finish_seconds)
    finish_minutes = finish_seconds / 60.0
    finish_hours = finish_seconds / 3600.0
    pace_per_km = calculate_pace_per_km(finish_seconds, MARATHON_DISTANCE_KM)
    pace_per_km_str = _format_pace(pace_per_km)
    pace_per_mile = calculate_pace_per_mile(finish_seconds, MARATHON_DISTANCE_KM)
    pace_per_mile_str = _format_pace(pace_per_mile)
    average_speed_kmh = calculate_average_speed_kmh(finish_seconds, MARATHON_DISTANCE_KM)
    average_speed_ms = average_speed_kmh / 3.6
    return {
        "finish_seconds": finish_seconds,
        "finish_time_str": finish_time_str,
        "finish_minutes": finish_minutes,
        "finish_hours": finish_hours,
        "pace_per_km": pace_per_km,
        "pace_per_km_str": pace_per_km_str,
        "pace_per_mile": pace_per_mile,
        "pace_per_mile_str": pace_per_mile_str,
        "average_speed_kmh": average_speed_kmh,
        "average_speed_ms": average_speed_ms,
    }


def _format_pace(pace_min_per_km: float) -> str:
    if pace_min_per_km <= 0:
        return "0:00"
    minutes = int(pace_min_per_km)
    seconds = round((pace_min_per_km - minutes) * 60)
    if seconds == 60:
        minutes += 1
        seconds = 0
    return f"{minutes}:{seconds:02d}"


def calculate_split_paces(split_seconds_dict: dict) -> dict:
    ordered_keys = ["5k", "10k", "15k", "20k", "half_marathon", "25k", "30k", "35k", "40k"]
    filtered_keys = [k for k in ordered_keys if k in split_seconds_dict]
    if not filtered_keys:
        return {}
    result = {}
    prev_distance = 0.0
    prev_time = 0.0
    for key in filtered_keys:
        current_distance = SPLIT_DISTANCES[key]
        current_time = split_seconds_dict[key]
        segment_distance = current_distance - prev_distance
        segment_time = current_time - prev_time
        if segment_distance > 0 and segment_time > 0:
            segment_pace = (segment_time / 60.0) / segment_distance
            label = f"{int(prev_distance)}_{key}"
            result[label] = round(segment_pace, 4)
        prev_distance = current_distance
        prev_time = current_time
    final_segment_distance = MARATHON_DISTANCE_KM - prev_distance
    if final_segment_distance > 0 and prev_time > 0:
        finish_time = split_seconds_dict.get("finish", None)
        if finish_time is not None and finish_time > prev_time:
            segment_time = finish_time - prev_time
            segment_pace = (segment_time / 60.0) / final_segment_distance
            label = f"{int(prev_distance)}_finish"
            result[label] = round(segment_pace, 4)
    return result


def calculate_first_second_half(finish_seconds: float, half_seconds: float) -> dict:
    first_half_seconds = half_seconds
    second_half_seconds = finish_seconds - half_seconds
    split_difference_seconds = second_half_seconds - first_half_seconds
    if abs(split_difference_seconds) < 1.0:
        split_type = "even"
    elif split_difference_seconds < 0:
        split_type = "negative"
    else:
        split_type = "positive"
    return {
        "first_half_seconds": first_half_seconds,
        "second_half_seconds": second_half_seconds,
        "split_type": split_type,
        "split_difference_seconds": split_difference_seconds,
    }


def calculate_pace_profile(split_paces: dict) -> dict:
    if not split_paces:
        return {
            "avg_pace": 0.0,
            "min_pace": 0.0,
            "max_pace": 0.0,
            "pace_std": 0.0,
            "pace_range": 0.0,
            "pace_trend": "even",
        }
    paces = list(split_paces.values())
    avg_pace = float(np.mean(paces))
    min_pace = float(np.min(paces))
    max_pace = float(np.max(paces))
    pace_std = float(np.std(paces, ddof=1)) if len(paces) > 1 else 0.0
    pace_range = max_pace - min_pace
    if len(paces) < 2:
        pace_trend = "even"
    else:
        first_half_paces = paces[: len(paces) // 2]
        second_half_paces = paces[len(paces) // 2 :]
        avg_first = float(np.mean(first_half_paces))
        avg_second = float(np.mean(second_half_paces))
        if abs(avg_second - avg_first) < 0.05:
            pace_trend = "even"
        elif avg_second < avg_first:
            pace_trend = "accelerating"
        else:
            pace_trend = "decelerating"
    return {
        "avg_pace": round(avg_pace, 4),
        "min_pace": round(min_pace, 4),
        "max_pace": round(max_pace, 4),
        "pace_std": round(pace_std, 4),
        "pace_range": round(pace_range, 4),
        "pace_trend": pace_trend,
    }


def identify_wall_segment(split_paces: dict) -> dict:
    if not split_paces:
        return {
            "wall_segment": "",
            "pace_before": 0.0,
            "pace_at_wall": 0.0,
            "pace_increase_pct": 0.0,
        }
    keys = list(split_paces.keys())
    if len(keys) < 2:
        return {
            "wall_segment": keys[0] if keys else "",
            "pace_before": split_paces.get(keys[0], 0.0) if keys else 0.0,
            "pace_at_wall": split_paces.get(keys[0], 0.0) if keys else 0.0,
            "pace_increase_pct": 0.0,
        }
    max_increase = 0.0
    wall_segment = keys[1]
    pace_before = split_paces[keys[0]]
    pace_at_wall = split_paces[keys[1]]
    for i in range(1, len(keys)):
        prev_pace = split_paces[keys[i - 1]]
        curr_pace = split_paces[keys[i]]
        if prev_pace > 0:
            increase_pct = ((curr_pace - prev_pace) / prev_pace) * 100.0
        else:
            increase_pct = 0.0
        if increase_pct > max_increase:
            max_increase = increase_pct
            wall_segment = keys[i]
            pace_before = prev_pace
            pace_at_wall = curr_pace
    return {
        "wall_segment": wall_segment,
        "pace_before": round(pace_before, 4),
        "pace_at_wall": round(pace_at_wall, 4),
        "pace_increase_pct": round(max_increase, 2),
    }


def calculate_even_pace_score(split_paces: dict) -> float:
    if not split_paces:
        return 0.0
    paces = list(split_paces.values())
    if len(paces) < 2:
        return 100.0
    mean_pace = np.mean(paces)
    if mean_pace <= 0:
        return 0.0
    std_pace = float(np.std(paces, ddof=1))
    cv = (std_pace / mean_pace) * 100.0
    score = max(0.0, min(100.0, 100.0 - cv * 10.0))
    return round(score, 2)


def estimate_vo2max(finish_seconds: float, age: int = 30, gender: str = "Male") -> float:
    finish_minutes = finish_seconds / 60.0
    finish_hours = finish_minutes / 60.0
    if finish_hours <= 0:
        return 0.0
    gender_normalized = gender.strip().lower() if gender else "male"
    if gender_normalized in ("female", "f", "women", "woman"):
        vo2max = 100.0 / finish_hours - 5.0
    else:
        vo2max = 110.0 / finish_hours - 5.0
    if age > 30:
        age_adjustment = (age - 30) * 0.5
        vo2max -= age_adjustment
    return round(max(0.0, vo2max), 2)


def calculate_training_paces(finish_seconds: float) -> dict:
    finish_pace = calculate_pace_per_km(finish_seconds, MARATHON_DISTANCE_KM)
    if finish_pace <= 0:
        return {
            "easy_pace": 0.0,
            "easy_pace_str": "0:00",
            "tempo_pace": 0.0,
            "tempo_pace_str": "0:00",
            "interval_pace": 0.0,
            "interval_pace_str": "0:00",
            "long_run_pace": 0.0,
            "long_run_pace_str": "0:00",
        }
    easy_pace = finish_pace * 1.275
    tempo_pace = finish_pace * 1.075
    interval_pace = finish_pace * 0.925
    long_run_pace = finish_pace * 1.35
    return {
        "easy_pace": round(easy_pace, 4),
        "easy_pace_str": _format_pace(easy_pace),
        "tempo_pace": round(tempo_pace, 4),
        "tempo_pace_str": _format_pace(tempo_pace),
        "interval_pace": round(interval_pace, 4),
        "interval_pace_str": _format_pace(interval_pace),
        "long_run_pace": round(long_run_pace, 4),
        "long_run_pace_str": _format_pace(long_run_pace),
    }


def age_graded_performance(finish_seconds: float, age: int, gender: str) -> dict:
    open_standards = {
        "male": 7200.0,
        "female": 8100.0,
    }
    age_factors = {
        "male": {
            20: 0.9985, 25: 1.0000, 30: 1.0000, 35: 0.9850,
            40: 0.9685, 45: 0.9490, 50: 0.9270, 55: 0.8960,
            60: 0.8650, 65: 0.8300, 70: 0.7920, 75: 0.7500,
            80: 0.7050,
        },
        "female": {
            20: 0.9985, 25: 1.0000, 30: 1.0000, 35: 0.9860,
            40: 0.9710, 45: 0.9530, 50: 0.9330, 55: 0.9050,
            60: 0.8770, 65: 0.8450, 70: 0.8100, 75: 0.7700,
            80: 0.7250,
        },
    }
    gender_normalized = gender.strip().lower() if gender else "male"
    if gender_normalized not in ("male", "female"):
        gender_normalized = "male"
    if gender_normalized in ("female", "f", "women", "woman"):
        gender_key = "female"
    else:
        gender_key = "male"
    open_standard = open_standards[gender_key]
    factors = age_factors[gender_key]
    if age <= 20:
        factor = factors[20]
    elif age >= 80:
        factor = factors[80]
    else:
        lower_age = max(k for k in factors.keys() if k <= age)
        upper_age = min(k for k in factors.keys() if k >= age)
        if lower_age == upper_age:
            factor = factors[lower_age]
        else:
            lower_factor = factors[lower_age]
            upper_factor = factors[upper_age]
            fraction = (age - lower_age) / (upper_age - lower_age)
            factor = lower_factor + fraction * (upper_factor - lower_factor)
    if factor > 0:
        adjusted_time = finish_seconds / factor
        graded_percentage = (open_standard / finish_seconds) * factor * 100.0
    else:
        adjusted_time = finish_seconds
        graded_percentage = 0.0
    if graded_percentage >= 90.0:
        performance_level = "World Class"
    elif graded_percentage >= 80.0:
        performance_level = "National Class"
    elif graded_percentage >= 70.0:
        performance_level = "Regional Class"
    elif graded_percentage >= 60.0:
        performance_level = "Local Class"
    else:
        performance_level = "Recreational"
    return {
        "graded_percentage": round(graded_percentage, 2),
        "adjusted_time": round(adjusted_time, 2),
        "performance_level": performance_level,
    }


def marathon_pace_chart(finish_seconds: float) -> pd.DataFrame:
    if finish_seconds <= 0:
        return pd.DataFrame(columns=["distance_km", "cumulative_time", "split_time", "pace_per_km"])
    avg_pace_per_km = finish_seconds / MARATHON_DISTANCE_KM
    split_distances_list = [5, 10, 15, 20, 25, 30, 35, 40, MARATHON_DISTANCE_KM]
    rows = []
    prev_distance = 0.0
    prev_cumulative_time = 0.0
    for dist in split_distances_list:
        cumulative_time = dist * avg_pace_per_km
        split_time = cumulative_time - prev_cumulative_time
        segment_distance = dist - prev_distance
        if segment_distance > 0:
            pace_per_km = (split_time / 60.0) / segment_distance
        else:
            pace_per_km = 0.0
        rows.append({
            "distance_km": round(dist, 3),
            "cumulative_time": seconds_to_time(cumulative_time),
            "split_time": seconds_to_time(split_time),
            "pace_per_km": round(pace_per_km, 4),
        })
        prev_distance = dist
        prev_cumulative_time = cumulative_time
    return pd.DataFrame(rows)
