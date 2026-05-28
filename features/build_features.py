import pandas as pd
import numpy as np
from src.utils.helpers import (
    time_to_seconds,
    seconds_to_time,
    calculate_pace_per_km,
    calculate_pace_per_mile,
    calculate_average_speed_kmh,
    calculate_average_speed_ms,
    categorize_performance,
    create_age_group,
    identify_negative_split,
    identify_positive_split,
    calculate_split_difference,
    calculate_pace_variation,
    validate_finish_time,
)


def build_running_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "finish_time" in df.columns:
        df["finish_seconds"] = df["finish_time"].apply(time_to_seconds)
    elif "finish_time_sec" in df.columns:
        df["finish_seconds"] = df["finish_time_sec"]
    else:
        df["finish_seconds"] = 0
    df["finish_minutes"] = df["finish_seconds"] / 60.0
    df["finish_hours"] = df["finish_seconds"] / 3600.0
    df["pace_per_km"] = df["finish_seconds"].apply(calculate_pace_per_km)
    df["pace_per_mile"] = df["finish_seconds"].apply(calculate_pace_per_mile)
    df["average_speed_kmh"] = df["finish_seconds"].apply(calculate_average_speed_kmh)
    df["average_speed_ms"] = df["finish_seconds"].apply(calculate_average_speed_ms)
    gender_col = "gender" if "gender" in df.columns else "sex"
    df["performance_category"] = df.apply(
        lambda row: categorize_performance(row["finish_seconds"], row.get(gender_col, "")),
        axis=1,
    )
    age_col = "age" if "age" in df.columns else None
    if age_col and age_col in df.columns:
        df["age_group"] = df[age_col].apply(lambda x: create_age_group(int(x)) if pd.notna(x) else "Unknown")
    else:
        df["age_group"] = "Unknown"
    return df


def build_split_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    split_cols = {
        "split_5k": 5,
        "split_10k": 10,
        "split_15k": 15,
        "split_20k": 20,
        "split_half_marathon": 21.0975,
        "split_25k": 25,
        "split_30k": 30,
        "split_35k": 35,
        "split_40k": 40,
    }
    for col, dist in split_cols.items():
        seconds_col = col + "_seconds"
        if col in df.columns:
            df[seconds_col] = df[col].apply(time_to_seconds)
        elif col + "_sec" in df.columns:
            df[seconds_col] = df[col + "_sec"]
        else:
            df[seconds_col] = 0
    if "split_half_marathon_seconds" in df.columns:
        df["first_half_seconds"] = df["split_half_marathon_seconds"]
    elif "split_half_sec" in df.columns:
        df["first_half_seconds"] = df["split_half_sec"]
    else:
        df["first_half_seconds"] = 0
    finish_col = "finish_seconds" if "finish_seconds" in df.columns else "finish_time_sec"
    if finish_col in df.columns:
        df["second_half_seconds"] = df[finish_col] - df["first_half_seconds"]
    else:
        df["second_half_seconds"] = 0
    df["negative_split_flag"] = df.apply(
        lambda row: int(identify_negative_split(row["first_half_seconds"], row["second_half_seconds"]))
        if row["first_half_seconds"] > 0 and row["second_half_seconds"] > 0 else 0,
        axis=1,
    )
    df["positive_split_flag"] = df.apply(
        lambda row: int(identify_positive_split(row["first_half_seconds"], row["second_half_seconds"]))
        if row["first_half_seconds"] > 0 and row["second_half_seconds"] > 0 else 0,
        axis=1,
    )
    df["split_difference_seconds"] = df.apply(
        lambda row: calculate_split_difference(row["first_half_seconds"], row["second_half_seconds"]),
        axis=1,
    )
    seg_distances = [5, 5, 5, 5, 5, 5, 5, 5, 2.195]
    seg_labels = [
        "pace_0_5k",
        "pace_5_10k",
        "pace_10_15k",
        "pace_15_20k",
        "pace_20_25k",
        "pace_25_30k",
        "pace_30_35k",
        "pace_35_40k",
        "pace_40_finish",
    ]
    cum_cols = [
        "split_5k_seconds",
        "split_10k_seconds",
        "split_15k_seconds",
        "split_20k_seconds",
        "split_25k_seconds",
        "split_30k_seconds",
        "split_35k_seconds",
        "split_40k_seconds",
        None,
    ]
    for i in range(len(seg_labels)):
        current_cum_col = cum_cols[i]
        if i == 0:
            prev_cum = 0
        else:
            prev_cum_col = cum_cols[i - 1]
            if prev_cum_col and prev_cum_col in df.columns:
                prev_cum = df[prev_cum_col]
            else:
                prev_cum = 0
        if current_cum_col and current_cum_col in df.columns:
            segment_time = df[current_cum_col] - prev_cum
        elif i == len(seg_labels) - 1 and finish_col in df.columns:
            last_split_col = cum_cols[i - 1]
            if last_split_col and last_split_col in df.columns:
                segment_time = df[finish_col] - df[last_split_col]
            else:
                segment_time = 0
        else:
            segment_time = 0
        dist_km = seg_distances[i]
        df[seg_labels[i]] = segment_time / dist_km / 60.0
    pace_cols = [col for col in seg_labels if col in df.columns]
    def row_pace_variation(row):
        paces = [row[col] for col in pace_cols if pd.notna(row[col]) and row[col] > 0]
        return calculate_pace_variation(paces)
    df["pace_variation"] = df.apply(row_pace_variation, axis=1)
    return df


def build_brazil_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    country_col = "country" if "country" in df.columns else "runner_country"
    if country_col in df.columns:
        df["is_brazilian"] = (df[country_col] == "Brazil").astype(int)
        df["is_brazilian"] = df["is_brazilian"].where(df[country_col].notna(), 0)
    else:
        df["is_brazilian"] = 0
    group_cols = []
    if "race_name" in df.columns:
        group_cols.append("race_name")
    elif "marathon" in df.columns:
        group_cols.append("marathon")
    if "year" in df.columns:
        group_cols.append("year")
    if group_cols:
        brazilian_share = df.groupby(group_cols)["is_brazilian"].mean().reset_index()
        brazilian_share.columns = group_cols + ["brazilian_share"]
        df = df.merge(brazilian_share, on=group_cols, how="left")
    else:
        df["brazilian_share"] = 0.0
    race_country_map = {
        "Tokyo": "Japan",
        "Boston": "USA",
        "London": "UK",
        "Berlin": "Germany",
        "Chicago": "USA",
        "New York City": "USA",
    }
    race_col = "race_name" if "race_name" in df.columns else "marathon"
    country_race_col = "country_held" if "country_held" in df.columns else None
    if country_race_col and country_race_col in df.columns:
        df["race_country"] = df[country_race_col]
    elif race_col in df.columns:
        df["race_country"] = df[race_col].map(race_country_map).fillna("")
    else:
        df["race_country"] = ""
    df["home_advantage"] = 0
    brazil_mask = df["is_brazilian"] == 1
    if country_col in df.columns:
        df.loc[brazil_mask, "home_advantage"] = (
            df.loc[brazil_mask, "race_country"].str.lower().isin(["brazil", "brasil"]).astype(int)
        )
    return df


def build_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = build_running_features(df)
    df = build_split_features(df)
    label_maps = {}
    cat_cols = []
    if "gender" in df.columns:
        cat_cols.append("gender")
    if "race_name" in df.columns:
        cat_cols.append("race_name")
    elif "marathon" in df.columns:
        cat_cols.append("marathon")
    if "performance_category" in df.columns:
        cat_cols.append("performance_category")
    for col in cat_cols:
        unique_vals = df[col].dropna().unique().tolist()
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        label_maps[col] = mapping
        df[col + "_encoded"] = df[col].map(mapping)
    gender_encoded_col = None
    if "gender_encoded" in df.columns:
        gender_encoded_col = "gender_encoded"
    if "age" in df.columns and gender_encoded_col:
        df["age_x_gender"] = df["age"] * df[gender_encoded_col]
    else:
        df["age_x_gender"] = 0
    if "pace_per_km" in df.columns and gender_encoded_col:
        df["pace_x_gender"] = df["pace_per_km"] * df[gender_encoded_col]
    else:
        df["pace_x_gender"] = 0.0
    if "age" in df.columns:
        df["age_squared"] = df["age"] ** 2
        df["age_cubed"] = df["age"] ** 3
    else:
        df["age_squared"] = 0
        df["age_cubed"] = 0
    if "finish_seconds" in df.columns:
        df = df.dropna(subset=["finish_seconds"])
    return df


def create_winners_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    gender_col = "gender" if "gender" in df.columns else "sex"
    race_col = "race_name" if "race_name" in df.columns else "marathon"
    finish_col = "finish_seconds" if "finish_seconds" in df.columns else "finish_time_sec"
    country_col = "country" if "country" in df.columns else "runner_country"
    name_col = "runner_name" if "runner_name" in df.columns else "name"
    if "overall_place" in df.columns:
        winners = df[df["overall_place"] == 1].copy()
    elif "place" in df.columns:
        winners = df[df["place"] == 1].copy()
    else:
        group_cols = [race_col, "year"]
        if gender_col in df.columns:
            group_cols.append(gender_col)
        valid = df[df[finish_col].notna() & (df[finish_col] > 0)].copy()
        idx = valid.groupby(group_cols)[finish_col].idxmin()
        winners = valid.loc[idx].copy()
    if "pace_per_km" not in winners.columns:
        winners["pace_per_km"] = winners[finish_col].apply(calculate_pace_per_km)
    if "average_speed_kmh" not in winners.columns:
        winners["average_speed_kmh"] = winners[finish_col].apply(calculate_average_speed_kmh)
    pivot_cols = {
        finish_col: ["winner_time_seconds_male", "winner_time_seconds_female"],
        "pace_per_km": ["winner_pace_per_km_male", "winner_pace_per_km_female"],
        "average_speed_kmh": ["winner_speed_kmh_male", "winner_speed_kmh_female"],
        country_col: ["winner_country_male", "winner_country_female"],
        name_col: ["winner_name_male", "winner_name_female"],
    }
    summary_frames = []
    group_cols_summary = [race_col, "year"]
    for gender_val, gender_label in [("M", "male"), ("F", "female")]:
        gender_winners = winners[winners[gender_col] == gender_val].copy()
        agg_dict = {}
        rename_dict = {}
        for src_col, col_pair in pivot_cols.items():
            if src_col in gender_winners.columns:
                target_col = col_pair[0] if gender_label == "male" else col_pair[1]
                agg_dict[src_col] = "first"
                rename_dict[src_col] = target_col
        if agg_dict:
            grouped = gender_winners.groupby(group_cols_summary).agg(agg_dict).reset_index()
            grouped = grouped.rename(columns=rename_dict)
            summary_frames.append(grouped)
    if summary_frames:
        from functools import reduce
        summary = reduce(lambda left, right: pd.merge(left, right, on=group_cols_summary, how="outer"), summary_frames)
    else:
        summary = pd.DataFrame(columns=group_cols_summary)
    return summary


def create_race_summary(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    race_col = "race_name" if "race_name" in df.columns else "marathon"
    finish_col = "finish_seconds" if "finish_seconds" in df.columns else "finish_time_sec"
    gender_col = "gender" if "gender" in df.columns else "sex"
    finished = df[df[finish_col].notna() & (df[finish_col] > 0)].copy()
    group_cols = [race_col, "year"]
    summary = finished.groupby(group_cols).agg(
        total_finishers=(finish_col, "count"),
        average_finish_time=(finish_col, "mean"),
        median_finish_time=(finish_col, "median"),
        fastest_time=(finish_col, "min"),
        slowest_time=(finish_col, "max"),
        std_finish_time=(finish_col, "std"),
    ).reset_index()
    summary["average_pace"] = summary["average_finish_time"].apply(calculate_pace_per_km)
    if gender_col in finished.columns:
        gender_dist = finished.groupby(group_cols)[gender_col].value_counts(normalize=True).unstack(fill_value=0).reset_index()
        gender_dist.columns = [str(c) for c in gender_dist.columns]
        gender_dist_dict = {}
        for _, row in gender_dist.iterrows():
            key = (row[race_col], row["year"])
            dist = {}
            for col in gender_dist.columns:
                if col not in group_cols:
                    dist[col] = row[col]
            gender_dist_dict[key] = dist
        summary["gender_distribution"] = summary.apply(
            lambda row: gender_dist_dict.get((row[race_col], row["year"]), {}),
            axis=1,
        )
    else:
        summary["gender_distribution"] = [{}] * len(summary)
    return summary
