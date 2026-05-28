import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

COLOR_PALETTE = {
    "background": "#F8FAFC",
    "primary": "#2563EB",
    "secondary": "#14B8A6",
    "accent": "#F97316",
    "success": "#22C55E",
    "warning": "#F59E0B",
    "text": "#0F172A",
    "muted": "#64748B",
    "card_bg": "#FFFFFF",
    "border": "#E2E8F0",
}

RACE_COLORS = {
    "Tokyo": "#E11D48",
    "Boston": "#2563EB",
    "London": "#7C3AED",
    "Berlin": "#14B8A6",
    "Chicago": "#F97316",
    "New York City": "#CA8A04",
}

PERFORMANCE_COLORS = {
    "Elite": "#2563EB",
    "Advanced": "#14B8A6",
    "Intermediate": "#F97316",
    "Recreational": "#64748B",
}


def create_kpi_card(title: str, value: str, delta: str = None) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Indicator(
            mode="number+delta" if delta else "number",
            value=None,
            title={"text": title, "font": {"size": 16, "color": COLOR_PALETTE["muted"]}},
            number={"font": {"size": 36, "color": COLOR_PALETTE["primary"]}, "textformat": None},
            delta={
                "reference": 0,
                "valueformat": ".1%",
                "font": {"size": 16},
                "decreasing": {"color": COLOR_PALETTE["accent"]},
                "increasing": {"color": COLOR_PALETTE["success"]},
            }
            if delta
            else None,
        )
    )
    fig.update_layout(
        annotations=[
            {
                "text": f"<b>{value}</b>",
                "x": 0.5,
                "y": 0.4,
                "font": {"size": 36, "color": COLOR_PALETTE["primary"]},
                "showarrow": False,
                "xanchor": "center",
            }
        ]
    )
    if delta:
        delta_color = COLOR_PALETTE["success"] if not delta.startswith("-") else COLOR_PALETTE["accent"]
        fig.add_annotation(
            text=delta,
            x=0.5,
            y=0.15,
            font={"size": 16, "color": delta_color},
            showarrow=False,
            xanchor="center",
        )
    fig.update_layout(
        paper_bgcolor=COLOR_PALETTE["card_bg"],
        plot_bgcolor=COLOR_PALETTE["card_bg"],
        margin=dict(l=20, r=20, t=50, b=20),
        height=160,
    )
    return fig


def plot_winner_time_evolution(df: pd.DataFrame, gender: str = "All") -> go.Figure:
    working = df.copy()
    if gender and gender != "All" and "gender" in working.columns:
        working = working[working["gender"] == gender]
    time_col = "winning_time_sec" if "winning_time_sec" in working.columns else "finish_time_sec"
    race_col = "marathon" if "marathon" in working.columns else "race"
    if "year" not in working.columns or time_col not in working.columns:
        return create_empty_figure("Missing required columns")
    races = working[race_col].dropna().unique()
    fig = go.Figure()
    for race in sorted(races):
        race_data = working[working[race_col] == race].sort_values("year")
        fig.add_trace(
            go.Scatter(
                x=race_data["year"],
                y=race_data[time_col] / 3600,
                mode="lines+markers",
                name=race,
                line=dict(color=RACE_COLORS.get(race, COLOR_PALETTE["primary"]), width=2),
                marker=dict(size=6),
            )
        )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Winning Time (hours)",
        legend_title="Race",
    )
    return apply_standard_layout(fig, title="Winning Time Evolution by Race")


def plot_race_comparison_bar(df: pd.DataFrame, metric: str = "finish_seconds") -> go.Figure:
    working = df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    metric_col = metric
    if metric == "finish_seconds" and "finish_time_sec" in working.columns:
        metric_col = "finish_time_sec"
    if "year" not in working.columns or race_col not in working.columns or metric_col not in working.columns:
        return create_empty_figure("Missing required columns")
    agg = working.groupby(["year", race_col])[metric_col].mean().reset_index()
    years = sorted(agg["year"].unique())
    races = sorted(agg[race_col].unique())
    fig = go.Figure()
    year_colors = px.colors.qualitative.Plotly
    for i, year in enumerate(years):
        year_data = agg[agg["year"] == year]
        values = []
        for race in races:
            match = year_data[year_data[race_col] == race]
            values.append(match[metric_col].values[0] if len(match) > 0 else 0)
        display_values = [v / 3600 if v > 1000 else v for v in values]
        fig.add_trace(
            go.Bar(
                name=str(year),
                x=races,
                y=display_values,
                marker_color=year_colors[i % len(year_colors)],
            )
        )
    y_label = "Average Finish Time (hours)" if metric_col in ("finish_time_sec", "finish_seconds") else metric
    fig.update_layout(
        xaxis_title="Race",
        yaxis_title=y_label,
        barmode="group",
        legend_title="Year",
    )
    return apply_standard_layout(fig, title="Race Comparison by Year")


def plot_finish_time_distribution(df: pd.DataFrame, gender: str = "All") -> go.Figure:
    working = df.copy()
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    if time_col not in working.columns:
        return create_empty_figure("Missing finish time column")
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    if gender and gender != "All" and "gender" in working.columns:
        working = working[working["gender"] == gender]
    times_hours = working[time_col] / 3600
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=times_hours,
            nbinsx=60,
            marker_color=COLOR_PALETTE["primary"],
            opacity=0.7,
            name="Finishers",
        )
    )
    mean_val = times_hours.mean()
    median_val = times_hours.median()
    fig.add_vline(
        x=mean_val,
        line_dash="dash",
        line_color=COLOR_PALETTE["accent"],
        annotation_text=f"Mean: {mean_val:.2f}h",
        annotation_position="top left",
    )
    fig.add_vline(
        x=median_val,
        line_dash="dot",
        line_color=COLOR_PALETTE["success"],
        annotation_text=f"Median: {median_val:.2f}h",
        annotation_position="top right",
    )
    fig.update_layout(
        xaxis_title="Finish Time (hours)",
        yaxis_title="Number of Finishers",
    )
    return apply_standard_layout(fig, title="Finish Time Distribution")


def plot_pace_distribution(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    pace_col = None
    for col in ["pace_per_km", "avg_pace_sec_per_km", "pace_per_km_sec"]:
        if col in working.columns:
            pace_col = col
            break
    if pace_col is None:
        time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
        if time_col in working.columns:
            working = working[working[time_col].notna() & (working[time_col] > 0)]
            working["pace_per_km"] = working[time_col] / 42.195
            pace_col = "pace_per_km"
        else:
            return create_empty_figure("Missing pace data")
    working = working[working[pace_col].notna() & (working[pace_col] > 0)]
    pace_min = working[pace_col] / 60
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=pace_min,
            nbinsx=50,
            marker_color=COLOR_PALETTE["secondary"],
            opacity=0.7,
            name="Runners",
        )
    )
    fig.update_layout(
        xaxis_title="Pace (min/km)",
        yaxis_title="Number of Runners",
    )
    return apply_standard_layout(fig, title="Pace Distribution")


def plot_age_distribution(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    if "age" not in working.columns:
        return create_empty_figure("Missing age column")
    working = working[working["age"].notna()]
    fig = go.Figure()
    fig.add_trace(
        go.Histogram(
            x=working["age"],
            nbinsx=40,
            marker_color=COLOR_PALETTE["primary"],
            opacity=0.7,
            name="Runners",
        )
    )
    boundaries = [18, 25, 35, 45, 55, 65]
    for boundary in boundaries:
        fig.add_vline(
            x=boundary,
            line_dash="dash",
            line_color=COLOR_PALETTE["muted"],
            opacity=0.5,
        )
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Number of Runners",
    )
    return apply_standard_layout(fig, title="Age Distribution")


def plot_country_bar(df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    working = df.copy()
    country_col = "country" if "country" in working.columns else "nationality"
    if country_col not in working.columns:
        return create_empty_figure("Missing country column")
    counts = working[country_col].value_counts().head(top_n).reset_index()
    counts.columns = ["country", "count"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts["country"],
            y=counts["count"],
            marker_color=COLOR_PALETTE["primary"],
            text=counts["count"],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Number of Finishers",
    )
    return apply_standard_layout(fig, title=f"Top {top_n} Countries by Finishers")


def plot_winners_by_country(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    country_col = "winner_country" if "winner_country" in working.columns else "country"
    if country_col not in working.columns:
        return create_empty_figure("Missing country column")
    counts = working[country_col].value_counts().reset_index()
    counts.columns = ["country", "wins"]
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=counts["country"],
            y=counts["wins"],
            marker_color=COLOR_PALETTE["accent"],
            text=counts["wins"],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Country",
        yaxis_title="Number of Wins",
    )
    return apply_standard_layout(fig, title="Race Wins by Country")


def plot_gender_comparison(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    gender_col = "gender" if "gender" in working.columns else "sex"
    race_col = "marathon" if "marathon" in working.columns else "race"
    required = [time_col, gender_col, race_col]
    for col in required:
        if col not in working.columns:
            return create_empty_figure(f"Missing {col} column")
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    agg = working.groupby([race_col, gender_col])[time_col].mean().reset_index()
    races = sorted(agg[race_col].unique())
    fig = go.Figure()
    for gender_label, color in [("M", COLOR_PALETTE["primary"]), ("F", COLOR_PALETTE["accent"])]:
        gender_data = agg[agg[gender_col] == gender_label]
        values = []
        for race in races:
            match = gender_data[gender_data[race_col] == race]
            values.append(match[time_col].values[0] / 3600 if len(match) > 0 else 0)
        fig.add_trace(
            go.Bar(
                name="Male" if gender_label == "M" else "Female",
                x=races,
                y=values,
                marker_color=color,
            )
        )
    fig.update_layout(
        xaxis_title="Race",
        yaxis_title="Average Finish Time (hours)",
        barmode="group",
        legend_title="Gender",
    )
    return apply_standard_layout(fig, title="Gender Comparison by Race")


def plot_covid_impact(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    count_col = "participants_estimate" if "participants_estimate" in working.columns else None
    if "year" not in working.columns or race_col not in working.columns:
        return create_empty_figure("Missing required columns")
    if count_col:
        agg = working.groupby(["year", race_col])[count_col].sum().reset_index()
    else:
        agg = working.groupby(["year", race_col]).size().reset_index(name="count")
        count_col = "count"
    total_by_year = agg.groupby("year")[count_col].sum().reset_index()
    races = sorted(agg[race_col].unique())
    fig = go.Figure()
    for race in races:
        race_data = agg[agg[race_col] == race].sort_values("year")
        fig.add_trace(
            go.Scatter(
                x=race_data["year"],
                y=race_data[count_col],
                mode="lines+markers",
                name=race,
                line=dict(color=RACE_COLORS.get(race, COLOR_PALETTE["primary"]), width=2),
                marker=dict(size=6),
            )
        )
    for yr in [2020, 2021]:
        fig.add_vrect(
            x0=yr - 0.4,
            x1=yr + 0.4,
            fillcolor=COLOR_PALETTE["warning"],
            opacity=0.12,
            line_width=0,
        )
    fig.add_annotation(
        x=2020.5,
        y=agg[count_col].max() if len(agg) > 0 else 0,
        text="COVID-19",
        font=dict(size=12, color=COLOR_PALETTE["warning"]),
        showarrow=False,
    )
    fig.update_layout(
        xaxis_title="Year",
        yaxis_title="Participants",
        legend_title="Race",
    )
    return apply_standard_layout(fig, title="COVID-19 Impact on Participation (2018-2025)")


def plot_brazil_performance(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    country_col = "country" if "country" in working.columns else "nationality"
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    required = [country_col, time_col]
    for col in required:
        if col not in working.columns:
            return create_empty_figure(f"Missing {col} column")
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    from plotly.subplots import make_subplots
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=("Finishers Over Time", "Pace Distribution", "Best Times"),
    )
    if "year" in working.columns:
        by_year = (
            working.groupby("year")
            .agg(count=(time_col, "size"), best=(time_col, "min"))
            .reset_index()
        )
        fig.add_trace(
            go.Bar(
                x=by_year["year"],
                y=by_year["count"],
                marker_color=COLOR_PALETTE["success"],
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=by_year["year"],
                y=by_year["best"] / 3600,
                marker_color=COLOR_PALETTE["primary"],
                showlegend=False,
            ),
            row=1,
            col=3,
        )
    pace_vals = working[time_col] / 42.195 / 60
    fig.add_trace(
        go.Histogram(
            x=pace_vals,
            nbinsx=30,
            marker_color=COLOR_PALETTE["secondary"],
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Year", row=1, col=1)
    fig.update_yaxes(title_text="Count", row=1, col=1)
    fig.update_xaxes(title_text="Pace (min/km)", row=1, col=2)
    fig.update_yaxes(title_text="Count", row=1, col=2)
    fig.update_xaxes(title_text="Year", row=1, col=3)
    fig.update_yaxes(title_text="Best Time (hours)", row=1, col=3)
    return apply_standard_layout(fig, title="Brazilian Runners Performance")


def plot_brazil_vs_world(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    country_col = "country" if "country" in working.columns else "nationality"
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    for col in [country_col, time_col]:
        if col not in working.columns:
            return create_empty_figure(f"Missing {col} column")
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    working["group"] = working[country_col].apply(
        lambda x: "Brazil" if x == "BRA" else "Rest of World"
    )
    fig = go.Figure()
    for group_name, color in [("Brazil", COLOR_PALETTE["success"]), ("Rest of World", COLOR_PALETTE["primary"])]:
        group_data = working[working["group"] == group_name][time_col] / 3600
        fig.add_trace(
            go.Box(
                y=group_data,
                name=group_name,
                marker_color=color,
            )
        )
    fig.update_layout(
        yaxis_title="Finish Time (hours)",
    )
    return apply_standard_layout(fig, title="Brazil vs Rest of World Finish Times")


def plot_split_progression(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    segment_col = "segment" if "segment" in working.columns else None
    pace_col = "pace_per_km_sec" if "pace_per_km_sec" in working.columns else "pace_per_km"
    perf_col = "performance_category" if "performance_category" in working.columns else None
    if segment_col is None or pace_col not in working.columns:
        return create_empty_figure("Missing segment or pace data")
    working = working[working[pace_col].notna() & (working[pace_col] > 0)]
    if perf_col is None:
        if "finish_time_sec" in working.columns:
            working["performance_category"] = working["finish_time_sec"].apply(
                lambda t: "Elite" if t < 8400 else ("Advanced" if t < 10200 else ("Intermediate" if t < 12600 else "Recreational"))
            )
            perf_col = "performance_category"
        else:
            agg = working.groupby(segment_col)[pace_col].mean().reset_index()
            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=agg[segment_col],
                    y=agg[pace_col] / 60,
                    mode="lines+markers",
                    marker_color=COLOR_PALETTE["primary"],
                )
            )
            fig.update_layout(xaxis_title="Segment", yaxis_title="Pace (min/km)")
            return apply_standard_layout(fig, title="Pace Progression by Segment")
    segment_order = ["5k", "10k", "15k", "20k", "half", "25k", "30k", "35k", "40k"]
    categories_order = ["Elite", "Advanced", "Intermediate", "Recreational"]
    present_segments = [s for s in segment_order if s in working[segment_col].unique()]
    present_categories = [c for c in categories_order if c in working[perf_col].unique()]
    agg = working.groupby([segment_col, perf_col])[pace_col].mean().reset_index()
    fig = go.Figure()
    for cat in present_categories:
        cat_data = agg[agg[perf_col] == cat]
        ordered_segments = [s for s in present_segments if s in cat_data[segment_col].values]
        y_vals = [cat_data[cat_data[segment_col] == s][pace_col].values[0] / 60 for s in ordered_segments]
        fig.add_trace(
            go.Scatter(
                x=ordered_segments,
                y=y_vals,
                mode="lines+markers",
                name=cat,
                line=dict(color=PERFORMANCE_COLORS.get(cat, COLOR_PALETTE["muted"]), width=2),
                marker=dict(size=6),
            )
        )
    fig.update_layout(
        xaxis_title="Segment",
        yaxis_title="Pace (min/km)",
        legend_title="Category",
    )
    return apply_standard_layout(fig, title="Pace Progression by Performance Category")


def plot_negative_positive_split(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    split_col = "split_type" if "split_type" in working.columns else None
    if split_col is None:
        half_col = "split_half_sec" if "split_half_sec" in working.columns else None
        time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
        if half_col and time_col in working.columns:
            working = working[working[time_col].notna() & working[half_col].notna() & (working[time_col] > 0)]
            working["second_half"] = working[time_col] - working[half_col]
            working["split_type"] = working.apply(
                lambda row: "Positive" if row["second_half"] > row[half_col]
                else ("Negative" if row["second_half"] < row[half_col] - 30 else "Even"),
                axis=1,
            )
            split_col = "split_type"
        else:
            return create_empty_figure("Missing split data")
    counts = working[split_col].value_counts().reset_index()
    counts.columns = ["split_type", "count"]
    colors = {"Positive": COLOR_PALETTE["accent"], "Negative": COLOR_PALETTE["success"], "Even": COLOR_PALETTE["primary"]}
    fig = go.Figure()
    fig.add_trace(
        go.Pie(
            labels=counts["split_type"],
            values=counts["count"],
            marker_colors=[colors.get(st, COLOR_PALETTE["muted"]) for st in counts["split_type"]],
            hole=0.45,
            textinfo="label+percent",
            textposition="outside",
        )
    )
    return apply_standard_layout(fig, title="Negative vs Positive Split Distribution")


def plot_pace_by_segment_heatmap(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    segment_col = "segment" if "segment" in working.columns else None
    pace_col = "pace_per_km_sec" if "pace_per_km_sec" in working.columns else "pace_per_km"
    race_col = "marathon" if "marathon" in working.columns else "race"
    if segment_col is None or pace_col not in working.columns or race_col not in working.columns:
        return create_empty_figure("Missing required columns")
    working = working[working[pace_col].notna() & (working[pace_col] > 0)]
    segment_order = ["5k", "10k", "15k", "20k", "half", "25k", "30k", "35k", "40k"]
    present_segments = [s for s in segment_order if s in working[segment_col].unique()]
    agg = working.groupby([race_col, segment_col])[pace_col].mean().reset_index()
    pivot = agg.pivot(index=race_col, columns=segment_col, values=pace_col)
    pivot = pivot[[c for c in present_segments if c in pivot.columns]]
    pivot = pivot / 60
    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            colorscale="YlOrRd",
            text=np.round(pivot.values, 2),
            texttemplate="%{text}",
            textfont={"size": 10},
            colorbar=dict(title="min/km"),
        )
    )
    fig.update_layout(
        xaxis_title="Segment",
        yaxis_title="Race",
    )
    return apply_standard_layout(fig, title="Average Pace by Segment and Race")


def plot_choropleth_map(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    country_col = "country" if "country" in working.columns else "nationality"
    if country_col not in working.columns:
        return create_empty_figure("Missing country column")
    counts = working[country_col].value_counts().reset_index()
    counts.columns = ["country", "count"]
    fig = go.Figure()
    fig.add_trace(
        go.Choropleth(
            locations=counts["country"],
            z=counts["count"],
            colorscale="Blues",
            colorbar=dict(title="Runners"),
            locationmode="ISO-3",
        )
    )
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type="natural earth",
        ),
    )
    return apply_standard_layout(fig, title="Runner Distribution by Country")


def plot_model_comparison(results: dict, metric: str) -> go.Figure:
    if not results:
        return create_empty_figure("No model results provided")
    model_names = list(results.keys())
    scores = []
    for name in model_names:
        val = results[name]
        if isinstance(val, dict):
            scores.append(val.get(metric, 0))
        else:
            scores.append(0)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=model_names,
            y=scores,
            marker_color=COLOR_PALETTE["primary"],
            text=[f"{s:.4f}" for s in scores],
            textposition="outside",
        )
    )
    fig.update_layout(
        xaxis_title="Model",
        yaxis_title=metric.replace("_", " ").title(),
    )
    return apply_standard_layout(fig, title=f"Model Comparison - {metric.replace('_', ' ').title()}")


def plot_feature_importance(importance_df: pd.DataFrame, top_n: int = 15) -> go.Figure:
    if importance_df.empty or len(importance_df.columns) < 2:
        return create_empty_figure("No feature importance data")
    feat_col = importance_df.columns[0]
    imp_col = importance_df.columns[1]
    top = importance_df.nlargest(top_n, imp_col)
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            y=top[feat_col],
            x=top[imp_col],
            orientation="h",
            marker_color=COLOR_PALETTE["secondary"],
            text=top[imp_col].round(4),
            textposition="outside",
        )
    )
    fig.update_layout(
        yaxis=dict(autorange="reversed"),
        xaxis_title="Importance",
        yaxis_title="Feature",
    )
    return apply_standard_layout(fig, title=f"Top {top_n} Feature Importance")


def plot_scatter_age_vs_time(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    if "age" not in working.columns or time_col not in working.columns:
        return create_empty_figure("Missing age or finish time columns")
    working = working[working["age"].notna() & working[time_col].notna() & (working[time_col] > 0)]
    sample = working.sample(n=min(5000, len(working)), random_state=42)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=sample["age"],
            y=sample[time_col] / 3600,
            mode="markers",
            marker=dict(
                color=COLOR_PALETTE["primary"],
                opacity=0.3,
                size=4,
            ),
            name="Runners",
        )
    )
    z = np.polyfit(working["age"], working[time_col] / 3600, 1)
    p = np.poly1d(z)
    x_line = np.linspace(working["age"].min(), working["age"].max(), 100)
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=p(x_line),
            mode="lines",
            line=dict(color=COLOR_PALETTE["accent"], width=2, dash="dash"),
            name="Trend",
        )
    )
    fig.update_layout(
        xaxis_title="Age",
        yaxis_title="Finish Time (hours)",
    )
    return apply_standard_layout(fig, title="Age vs Finish Time")


def plot_boxplot_times_by_race(df: pd.DataFrame) -> go.Figure:
    working = df.copy()
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    race_col = "marathon" if "marathon" in working.columns else "race"
    for col in [time_col, race_col]:
        if col not in working.columns:
            return create_empty_figure(f"Missing {col} column")
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    races = sorted(working[race_col].unique())
    fig = go.Figure()
    for race in races:
        race_data = working[working[race_col] == race][time_col] / 3600
        fig.add_trace(
            go.Box(
                y=race_data,
                name=race,
                marker_color=RACE_COLORS.get(race, COLOR_PALETTE["primary"]),
            )
        )
    fig.update_layout(
        yaxis_title="Finish Time (hours)",
        showlegend=False,
    )
    return apply_standard_layout(fig, title="Finish Time Distribution by Race")


def create_empty_figure(message: str = "No data available") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.5,
        font=dict(size=18, color=COLOR_PALETTE["muted"]),
        showarrow=False,
        xanchor="center",
        yanchor="middle",
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        paper_bgcolor=COLOR_PALETTE["card_bg"],
        plot_bgcolor=COLOR_PALETTE["card_bg"],
        margin=dict(l=20, r=20, t=40, b=20),
        height=400,
    )
    return fig


def apply_standard_layout(fig: go.Figure, title: str = "") -> go.Figure:
    fig.update_layout(
        title=dict(
            text=title,
            font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
            x=0.5,
            xanchor="center",
        )
        if title
        else None,
        template="plotly_white",
        paper_bgcolor=COLOR_PALETTE["card_bg"],
        plot_bgcolor=COLOR_PALETTE["background"],
        font=dict(family="Arial", size=12, color=COLOR_PALETTE["text"]),
        margin=dict(l=60, r=30, t=60, b=50),
        xaxis=dict(
            gridcolor=COLOR_PALETTE["border"],
            linecolor=COLOR_PALETTE["border"],
            tickfont=dict(color=COLOR_PALETTE["muted"]),
            title_font=dict(color=COLOR_PALETTE["text"]),
        ),
        yaxis=dict(
            gridcolor=COLOR_PALETTE["border"],
            linecolor=COLOR_PALETTE["border"],
            tickfont=dict(color=COLOR_PALETTE["muted"]),
            title_font=dict(color=COLOR_PALETTE["text"]),
        ),
        legend=dict(
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor=COLOR_PALETTE["border"],
            borderwidth=1,
            font=dict(color=COLOR_PALETTE["text"]),
        ),
        hoverlabel=dict(
            bgcolor=COLOR_PALETTE["card_bg"],
            font_size=12,
            font_color=COLOR_PALETTE["text"],
        ),
    )
    return fig
