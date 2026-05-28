import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_utils import load_css

load_css()

from src.visualization.plots import (
    COLOR_PALETTE,
    RACE_COLORS,
    PERFORMANCE_COLORS,
    create_empty_figure,
    apply_standard_layout,
)

from src.utils.config import (
    MARATHON_DISTANCE_KM,
    RACE_NAMES,
    YEARS,
    PERFORMANCE_CATEGORIES,
)

from src.utils.helpers import (
    seconds_to_time,
    calculate_pace_per_km,
    format_pace_str,
    categorize_performance,
)


@st.cache_data
def compute_pace_overview_metrics(results_df, pace_splits_df):
    metrics = {}
    if results_df is None or results_df.empty:
        return metrics

    time_col = "finish_time_sec" if "finish_time_sec" in results_df.columns else "finish_seconds"
    gender_col = "gender" if "gender" in results_df.columns else "sex"
    finished = results_df[results_df["status"] == "Finished"] if "status" in results_df.columns else results_df
    finished = finished[finished[time_col].notna() & (finished[time_col] > 0)].copy()

    if finished.empty:
        return metrics

    finished["pace_per_km"] = finished[time_col] / MARATHON_DISTANCE_KM
    avg_pace_sec = finished["pace_per_km"].mean()
    avg_speed = MARATHON_DISTANCE_KM / (finished[time_col].mean() / 3600)
    metrics["avg_pace_sec"] = avg_pace_sec
    metrics["avg_speed_kmh"] = avg_speed

    finished["performance_category"] = finished.apply(
        lambda row: categorize_performance(row[time_col], row[gender_col]) if gender_col in finished.columns else "Recreational",
        axis=1,
    )

    elite = finished[finished["performance_category"] == "Elite"]
    recreational = finished[finished["performance_category"] == "Recreational"]
    metrics["fastest_avg_pace_sec"] = elite["pace_per_km"].min() if len(elite) > 0 else avg_pace_sec
    metrics["slowest_avg_pace_sec"] = recreational["pace_per_km"].max() if len(recreational) > 0 else avg_pace_sec

    if pace_splits_df is not None and not pace_splits_df.empty:
        pace_col = "pace_per_km_sec" if "pace_per_km_sec" in pace_splits_df.columns else "pace_per_km"
        pace_data = pace_splits_df[pace_splits_df[pace_col].notna() & (pace_splits_df[pace_col] > 0)]
        if not pace_data.empty:
            runner_pace_std = pace_data.groupby("bib_number")[pace_col].std()
            metrics["pace_variation"] = runner_pace_std.mean()
        else:
            metrics["pace_variation"] = 0.0
    else:
        metrics["pace_variation"] = 0.0

    split_col = "split_type" if "split_type" in results_df.columns else None
    if split_col and split_col in results_df.columns:
        split_valid = results_df[results_df[split_col].notna()]
        if len(split_valid) > 0:
            positive_count = len(split_valid[split_valid[split_col] == "Positive"])
            metrics["pct_positive_splits"] = (positive_count / len(split_valid)) * 100
        else:
            metrics["pct_positive_splits"] = 0.0
    else:
        metrics["pct_positive_splits"] = 0.0

    return metrics


@st.cache_data
def compute_split_progression(pace_splits_df, perf_categories):
    if pace_splits_df is None or pace_splits_df.empty:
        return pd.DataFrame()

    pace_col = "pace_per_km_sec" if "pace_per_km_sec" in pace_splits_df.columns else "pace_per_km"
    segment_col = "segment" if "segment" in pace_splits_df.columns else None
    time_col = "finish_time_sec" if "finish_time_sec" in pace_splits_df.columns else "finish_seconds"
    gender_col = "gender" if "gender" in pace_splits_df.columns else "sex"

    if segment_col is None or pace_col not in pace_splits_df.columns:
        return pd.DataFrame()

    working = pace_splits_df[pace_splits_df[pace_col].notna() & (pace_splits_df[pace_col] > 0)].copy()

    if time_col in working.columns and gender_col in working.columns:
        working["performance_category"] = working.apply(
            lambda row: categorize_performance(row[time_col], row[gender_col]),
            axis=1,
        )
    elif time_col in working.columns:
        working["performance_category"] = working[time_col].apply(
            lambda t: "Elite" if t < 8400 else ("Advanced" if t < 10200 else ("Intermediate" if t < 12600 else "Recreational"))
        )
    else:
        return pd.DataFrame()

    working = working[working["performance_category"].isin(perf_categories)]

    segment_order = ["5k", "10k", "15k", "20k", "half", "25k", "30k", "35k", "40k"]
    segment_labels = {
        "5k": "0-5 km",
        "10k": "5-10 km",
        "15k": "10-15 km",
        "20k": "15-20 km",
        "half": "Meia",
        "25k": "20-25 km",
        "30k": "25-30 km",
        "35k": "30-35 km",
        "40k": "35-40 km",
    }

    agg = working.groupby(["segment", "performance_category"])[pace_col].mean().reset_index()
    agg["segment_label"] = agg["segment"].map(segment_labels)
    agg["pace_min_km"] = agg[pace_col] / 60.0

    present_segments = [s for s in segment_order if s in agg["segment"].unique()]
    agg = agg[agg["segment"].isin(present_segments)]
    agg["segment_order"] = agg["segment"].apply(lambda s: segment_order.index(s) if s in segment_order else 99)
    agg = agg.sort_values(["segment_order", "performance_category"])

    return agg


@st.cache_data
def compute_negative_positive_split(results_df, perf_categories):
    if results_df is None or results_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    time_col = "finish_time_sec" if "finish_time_sec" in results_df.columns else "finish_seconds"
    gender_col = "gender" if "gender" in results_df.columns else "sex"
    half_col = "split_half_sec" if "split_half_sec" in results_df.columns else None

    finished = results_df.copy()
    if "status" in finished.columns:
        finished = finished[finished["status"] == "Finished"]

    finished = finished[finished[time_col].notna() & (finished[time_col] > 0)].copy()

    split_col = "split_type" if "split_type" in finished.columns else None
    if split_col is None or split_col not in finished.columns:
        if half_col and half_col in finished.columns:
            finished = finished[finished[half_col].notna() & (finished[half_col] > 0)].copy()
            finished["second_half_sec"] = finished[time_col] - finished[half_col]
            finished["split_type"] = finished.apply(
                lambda row: "Positivo" if row["second_half_sec"] > row[half_col]
                else ("Negativo" if row["second_half_sec"] < row[half_col] - 30 else "Uniforme"),
                axis=1,
            )
            split_col = "split_type"
        else:
            return pd.DataFrame(), pd.DataFrame()

    finished = finished[finished[split_col].notna()]
    split_counts = finished[split_col].value_counts().reset_index()
    split_counts.columns = ["split_type", "count"]

    if gender_col in finished.columns and time_col in finished.columns:
        finished["performance_category"] = finished.apply(
            lambda row: categorize_performance(row[time_col], row[gender_col]),
            axis=1,
        )
    elif time_col in finished.columns:
        finished["performance_category"] = finished[time_col].apply(
            lambda t: "Elite" if t < 8400 else ("Advanced" if t < 10200 else ("Intermediate" if t < 12600 else "Recreational"))
        )

    finished = finished[finished["performance_category"].isin(perf_categories)]
    by_cat = finished.groupby("performance_category")[split_col].apply(
        lambda x: (x == "Negativo").sum() / len(x) * 100 if len(x) > 0 else 0
    ).reset_index()
    by_cat.columns = ["performance_category", "pct_negative"]

    return split_counts, by_cat


@st.cache_data
def compute_segment_heatmap(pace_splits_df, selected_races):
    if pace_splits_df is None or pace_splits_df.empty:
        return pd.DataFrame()

    pace_col = "pace_per_km_sec" if "pace_per_km_sec" in pace_splits_df.columns else "pace_per_km"
    segment_col = "segment" if "segment" in pace_splits_df.columns else None
    race_col = "marathon" if "marathon" in pace_splits_df.columns else "race"

    if segment_col is None or pace_col not in pace_splits_df.columns or race_col not in pace_splits_df.columns:
        return pd.DataFrame()

    working = pace_splits_df[
        pace_splits_df[pace_col].notna() & (pace_splits_df[pace_col] > 0)
    ].copy()

    if selected_races:
        working = working[working[race_col].isin(selected_races)]

    segment_order = ["5k", "10k", "15k", "20k", "half", "25k", "30k", "35k", "40k"]
    segment_labels = {
        "5k": "0-5 km",
        "10k": "5-10 km",
        "15k": "10-15 km",
        "20k": "15-20 km",
        "half": "Meia",
        "25k": "20-25 km",
        "30k": "25-30 km",
        "35k": "30-35 km",
        "40k": "35-40 km",
    }

    agg = working.groupby([race_col, segment_col])[pace_col].mean().reset_index()
    pivot = agg.pivot(index=race_col, columns=segment_col, values=pace_col)
    present_segments = [s for s in segment_order if s in pivot.columns]
    pivot = pivot[[c for c in present_segments if c in pivot.columns]]
    pivot.columns = [segment_labels.get(c, c) for c in pivot.columns]
    pivot = pivot / 60.0

    return pivot


@st.cache_data
def compute_pace_dropoff(results_df, selected_races):
    if results_df is None or results_df.empty:
        return pd.DataFrame()

    time_col = "finish_time_sec" if "finish_time_sec" in results_df.columns else "finish_seconds"
    race_col = "marathon" if "marathon" in results_df.columns else "race"
    half_col = "split_half_sec" if "split_half_sec" in results_df.columns else None

    if time_col not in results_df.columns or half_col is None or half_col not in results_df.columns:
        return pd.DataFrame()

    finished = results_df.copy()
    if "status" in finished.columns:
        finished = finished[finished["status"] == "Finished"]

    finished = finished[
        finished[time_col].notna() & finished[half_col].notna() &
        (finished[time_col] > 0) & (finished[half_col] > 0)
    ].copy()

    if selected_races and race_col in finished.columns:
        finished = finished[finished[race_col].isin(selected_races)]

    finished["first_half_pace"] = finished[half_col] / 21.0975
    finished["second_half_pace"] = (finished[time_col] - finished[half_col]) / (MARATHON_DISTANCE_KM - 21.0975)
    finished["pace_increase_pct"] = ((finished["second_half_pace"] - finished["first_half_pace"]) / finished["first_half_pace"]) * 100

    if race_col not in finished.columns:
        return pd.DataFrame()

    dropoff = finished.groupby(race_col)["pace_increase_pct"].mean().reset_index()
    dropoff.columns = ["race", "pace_increase_pct"]
    dropoff = dropoff.sort_values("pace_increase_pct", ascending=False)

    return dropoff


@st.cache_data
def compute_segment_stats_table(pace_splits_df, perf_categories):
    if pace_splits_df is None or pace_splits_df.empty:
        return pd.DataFrame()

    pace_col = "pace_per_km_sec" if "pace_per_km_sec" in pace_splits_df.columns else "pace_per_km"
    segment_col = "segment" if "segment" in pace_splits_df.columns else None
    time_col = "finish_time_sec" if "finish_time_sec" in pace_splits_df.columns else "finish_seconds"
    gender_col = "gender" if "gender" in pace_splits_df.columns else "sex"

    if segment_col is None or pace_col not in pace_splits_df.columns:
        return pd.DataFrame()

    working = pace_splits_df[
        pace_splits_df[pace_col].notna() & (pace_splits_df[pace_col] > 0)
    ].copy()

    if time_col in working.columns and gender_col in working.columns:
        working["performance_category"] = working.apply(
            lambda row: categorize_performance(row[time_col], row[gender_col]),
            axis=1,
        )
    elif time_col in working.columns:
        working["performance_category"] = working[time_col].apply(
            lambda t: "Elite" if t < 8400 else ("Advanced" if t < 10200 else ("Intermediate" if t < 12600 else "Recreational"))
        )
    else:
        return pd.DataFrame()

    working = working[working["performance_category"].isin(perf_categories)]

    segment_order = ["5k", "10k", "15k", "20k", "half", "25k", "30k", "35k", "40k"]
    segment_labels = {
        "5k": "0-5 km",
        "10k": "5-10 km",
        "15k": "10-15 km",
        "20k": "15-20 km",
        "half": "Meia",
        "25k": "20-25 km",
        "30k": "25-30 km",
        "35k": "30-35 km",
        "40k": "35-40 km",
    }

    agg = working.groupby(["performance_category", "segment"])[pace_col].mean().reset_index()
    agg["pace_min_km"] = agg[pace_col] / 60.0
    agg["segment_label"] = agg["segment"].map(segment_labels)

    pivot = agg.pivot(index="performance_category", columns="segment", values="pace_min_km")
    present_segments = [s for s in segment_order if s in pivot.columns]
    pivot = pivot[[c for c in present_segments if c in pivot.columns]]
    pivot.columns = [segment_labels.get(c, c) for c in pivot.columns]
    pivot.index.name = "Categoria"
    pivot = pivot.reset_index()

    cat_order = [c for c in ["Elite", "Advanced", "Intermediate", "Recreational"] if c in pivot["Categoria"].values]
    pivot["sort_key"] = pivot["Categoria"].apply(lambda c: cat_order.index(c) if c in cat_order else 99)
    pivot = pivot.sort_values("sort_key").drop(columns="sort_key")

    return pivot


@st.cache_data
def compute_even_pace_runners(pace_splits_df):
    if pace_splits_df is None or pace_splits_df.empty:
        return pd.DataFrame()

    pace_col = "pace_per_km_sec" if "pace_per_km_sec" in pace_splits_df.columns else "pace_per_km"
    time_col = "finish_time_sec" if "finish_time_sec" in pace_splits_df.columns else "finish_seconds"
    gender_col = "gender" if "gender" in pace_splits_df.columns else "sex"

    if pace_col not in pace_splits_df.columns:
        return pd.DataFrame()

    working = pace_splits_df[
        pace_splits_df[pace_col].notna() & (pace_splits_df[pace_col] > 0)
    ].copy()

    runner_pace_std = working.groupby("bib_number")[pace_col].std().reset_index()
    runner_pace_std.columns = ["bib_number", "pace_std"]

    runner_info = working.groupby("bib_number").agg(
        runner_name=("runner_name", "first"),
        marathon=("marathon", "first"),
        year=("year", "first"),
    ).reset_index()

    runner_stats = runner_info.merge(runner_pace_std, on="bib_number", how="left")

    if time_col in working.columns:
        finish_times = working.groupby("bib_number")[time_col].first().reset_index()
        finish_times.columns = ["bib_number", "finish_time_sec"]
        runner_stats = runner_stats.merge(finish_times, on="bib_number", how="left")

    if gender_col in working.columns:
        genders = working.groupby("bib_number")[gender_col].first().reset_index()
        genders.columns = ["bib_number", "gender"]
        runner_stats = runner_stats.merge(genders, on="bib_number", how="left")

    runner_stats = runner_stats[runner_stats["pace_std"].notna()]
    runner_stats = runner_stats.sort_values("pace_std", ascending=True)
    runner_stats = runner_stats.head(20)

    if "finish_time_sec" in runner_stats.columns:
        runner_stats["finish_time"] = runner_stats["finish_time_sec"].apply(
            lambda s: seconds_to_time(s) if pd.notna(s) and s > 0 else "N/A"
        )

    return runner_stats


def render_pace_splits_page(results_df, pace_splits_df):
    st.title("Analise de Ritmo e Parciais")
    st.markdown(
        "Esta pagina examina como os corredores distribuem seu ritmo ao longo dos 42,195 km "
        "da maratona. A analise das parciais a cada 5 km revela como a fadiga, o terreno e a "
        "estrategia afetam o ritmo ao longo da prova, incluindo o fenomeno conhecido como "
        "'The Wall' (o muro) por volta dos 30-35 km."
    )
    st.divider()

    with st.expander("Filtros", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)

        race_col = "marathon" if results_df is not None and "marathon" in results_df.columns else "race"
        available_races = sorted(results_df[race_col].dropna().unique().tolist()) if results_df is not None and race_col in results_df.columns else []
        default_races = available_races if len(available_races) > 0 else []

        with filter_col1:
            selected_races = st.multiselect(
                "Corrida",
                options=available_races,
                default=default_races,
                key="pace_splits_race_filter",
            )
        with filter_col2:
            selected_categories = st.multiselect(
                "Categoria de Desempenho",
                options=["Elite", "Advanced", "Intermediate", "Recreational"],
                default=["Elite", "Advanced", "Intermediate", "Recreational"],
                key="pace_splits_perf_filter",
            )
        with filter_col3:
            selected_gender = st.selectbox(
                "Genero",
                options=["Todos", "Masculino", "Feminino"],
                index=0,
                key="pace_splits_gender_filter",
            )

    filtered_results = results_df.copy() if results_df is not None else pd.DataFrame()
    filtered_splits = pace_splits_df.copy() if pace_splits_df is not None else pd.DataFrame()

    if not filtered_results.empty and selected_races and race_col in filtered_results.columns:
        filtered_results = filtered_results[filtered_results[race_col].isin(selected_races)]

    if not filtered_splits.empty and selected_races:
        splits_race_col = "marathon" if "marathon" in filtered_splits.columns else "race"
        if splits_race_col in filtered_splits.columns:
            filtered_splits = filtered_splits[filtered_splits[splits_race_col].isin(selected_races)]

    if selected_gender != "Todos" and not filtered_results.empty:
        gender_col = "gender" if "gender" in filtered_results.columns else "sex"
        gender_val = "M" if selected_gender == "Masculino" else "F"
        if gender_col in filtered_results.columns:
            filtered_results = filtered_results[filtered_results[gender_col] == gender_val]

    if selected_gender != "Todos" and not filtered_splits.empty:
        splits_gender_col = "gender" if "gender" in filtered_splits.columns else "sex"
        gender_val = "M" if selected_gender == "Masculino" else "F"
        if splits_gender_col in filtered_splits.columns:
            filtered_splits = filtered_splits[filtered_splits[splits_gender_col] == gender_val]

    st.markdown("### Visao Geral do Ritmo")

    overview_metrics = compute_pace_overview_metrics(filtered_results, filtered_splits)

    kpi_row1_col1, kpi_row1_col2, kpi_row1_col3 = st.columns(3)
    with kpi_row1_col1:
        avg_pace = overview_metrics.get("avg_pace_sec", 0)
        avg_pace_min = avg_pace / 60.0
        st.metric(
            label="Ritmo Medio",
            value=f"{format_pace_str(avg_pace_min)} /km",
        )
    with kpi_row1_col2:
        fastest_pace = overview_metrics.get("fastest_avg_pace_sec", 0)
        fastest_pace_min = fastest_pace / 60.0
        st.metric(
            label="Ritmo Medio Mais Rapido (Elite)",
            value=f"{format_pace_str(fastest_pace_min)} /km",
        )
    with kpi_row1_col3:
        slowest_pace = overview_metrics.get("slowest_avg_pace_sec", 0)
        slowest_pace_min = slowest_pace / 60.0
        st.metric(
            label="Ritmo Medio Mais Lento (Recreativo)",
            value=f"{format_pace_str(slowest_pace_min)} /km",
        )

    st.markdown("")
    kpi_row2_col1, kpi_row2_col2, kpi_row2_col3 = st.columns(3)
    with kpi_row2_col1:
        avg_speed = overview_metrics.get("avg_speed_kmh", 0)
        st.metric(
            label="Velocidade Media",
            value=f"{avg_speed:.1f} km/h",
        )
    with kpi_row2_col2:
        pace_var = overview_metrics.get("pace_variation", 0)
        pace_var_min = pace_var / 60.0 if pace_var > 0 else 0
        st.metric(
            label="Variacao de Ritmo",
            value=f"{pace_var_min:.2f} min/km",
        )
    with kpi_row2_col3:
        pct_positive = overview_metrics.get("pct_positive_splits", 0)
        st.metric(
            label="% Parciais Positivas",
            value=f"{pct_positive:.1f}%",
        )

    st.divider()

    st.markdown("### Distribuicao do Ritmo")
    dist_col1, dist_col2 = st.columns(2)

    with dist_col1:
        st.markdown("#### Distribuicao Geral do Ritmo")
        if filtered_results is not None and not filtered_results.empty:
            time_col = "finish_time_sec" if "finish_time_sec" in filtered_results.columns else "finish_seconds"
            pace_data = filtered_results[filtered_results[time_col].notna() & (filtered_results[time_col] > 0)].copy()
            if not pace_data.empty:
                pace_data["pace_per_km"] = pace_data[time_col] / MARATHON_DISTANCE_KM
                pace_min = pace_data["pace_per_km"] / 60.0
                fig_hist = px.histogram(
                    pace_min,
                    x="pace_per_km",
                    nbins=50,
                    labels={"pace_per_km": "Ritmo (min/km)"},
                    color_discrete_sequence=[COLOR_PALETTE["primary"]],
                    opacity=0.7,
                )
                fig_hist.update_layout(
                    xaxis_title="Ritmo (min/km)",
                    yaxis_title="Numero de Corredores",
                    showlegend=False,
                )
                fig_hist = apply_standard_layout(fig_hist, title="Distribuicao do Ritmo (Todos os Corredores)")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.plotly_chart(create_empty_figure("Sem dados de ritmo disponiveis"), use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados de resultados disponiveis"), use_container_width=True)

    with dist_col2:
        st.markdown("#### Ritmo por Categoria de Desempenho")
        if filtered_results is not None and not filtered_results.empty:
            time_col = "finish_time_sec" if "finish_time_sec" in filtered_results.columns else "finish_seconds"
            gender_col = "gender" if "gender" in filtered_results.columns else "sex"
            pace_data = filtered_results[filtered_results[time_col].notna() & (filtered_results[time_col] > 0)].copy()
            if not pace_data.empty:
                pace_data["pace_per_km"] = pace_data[time_col] / MARATHON_DISTANCE_KM / 60.0
                if gender_col in pace_data.columns:
                    pace_data["performance_category"] = pace_data.apply(
                        lambda row: categorize_performance(row[time_col], row[gender_col]),
                        axis=1,
                    )
                else:
                    pace_data["performance_category"] = pace_data[time_col].apply(
                        lambda t: "Elite" if t < 8400 else ("Advanced" if t < 10200 else ("Intermediate" if t < 12600 else "Recreational"))
                    )
                pace_data = pace_data[pace_data["performance_category"].isin(selected_categories)]
                fig_cat_hist = px.histogram(
                    pace_data,
                    x="pace_per_km",
                    color="performance_category",
                    nbins=50,
                    barmode="overlay",
                    labels={"pace_per_km": "Ritmo (min/km)", "performance_category": "Categoria"},
                    opacity=0.6,
                    color_discrete_map=PERFORMANCE_COLORS,
                    category_orders={"performance_category": ["Elite", "Advanced", "Intermediate", "Recreational"]},
                )
                fig_cat_hist.update_layout(
                    xaxis_title="Ritmo (min/km)",
                    yaxis_title="Numero de Corredores",
                    legend_title="Categoria",
                )
                fig_cat_hist = apply_standard_layout(fig_cat_hist, title="Distribuicao do Ritmo por Categoria")
                st.plotly_chart(fig_cat_hist, use_container_width=True)
            else:
                st.plotly_chart(create_empty_figure("Sem dados de ritmo disponiveis"), use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados de resultados disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Progressao das Parciais por Categoria")
    if not filtered_splits.empty and selected_categories:
        progression_data = compute_split_progression(filtered_splits, selected_categories)
        if not progression_data.empty:
            fig_progression = go.Figure()
            cat_order = [c for c in ["Elite", "Advanced", "Intermediate", "Recreational"] if c in progression_data["performance_category"].unique()]
            for cat in cat_order:
                cat_data = progression_data[progression_data["performance_category"] == cat].sort_values("segment_order")
                fig_progression.add_trace(
                    go.Scatter(
                        x=cat_data["segment_label"],
                        y=cat_data["pace_min_km"],
                        mode="lines+markers",
                        name=cat,
                        line=dict(color=PERFORMANCE_COLORS.get(cat, COLOR_PALETTE["muted"]), width=2),
                        marker=dict(size=7),
                    )
                )

            segment_order_list = progression_data.drop_duplicates("segment")[["segment", "segment_label", "segment_order"]].sort_values("segment_order")
            wall_idx = None
            for idx_val, (_, row) in enumerate(segment_order_list.iterrows()):
                if row["segment"] == "35k":
                    wall_idx = idx_val
                    break

            if wall_idx is not None:
                fig_progression.add_vrect(
                    x0=wall_idx - 0.5,
                    x1=wall_idx + 0.5,
                    fillcolor="#FEE2E2",
                    opacity=0.3,
                    line_width=0,
                )
                fig_progression.add_annotation(
                    x=wall_idx,
                    y=fig_progression.data[0].y.max() if len(fig_progression.data) > 0 else 0,
                    text="The Wall",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(size=12, color="#DC2626"),
                    arrowcolor="#DC2626",
                    ay=-30,
                )

            fig_progression.update_layout(
                xaxis_title="Segmento",
                yaxis_title="Ritmo (min/km)",
                legend_title="Categoria",
            )
            fig_progression = apply_standard_layout(fig_progression, title="Progressao do Ritmo por Segmentos de 5 km")
            st.plotly_chart(fig_progression, use_container_width=True)
            st.markdown(
                "A zona destacada marca o segmento de 30-35 km onde os corredores normalmente "
                "experimentam a maior desaceleracao de ritmo, fenomeno conhecido como 'The Wall' "
                "(o muro). Isso ocorre quando as reservas de glicogenio se esgotam e a fadiga "
                "muscular se acumula de forma significativa."
            )
        else:
            st.plotly_chart(create_empty_figure("Dados insuficientes para progressao das parciais"), use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("Sem dados de parciais disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Analise de Parciais Negativas vs Positivas")
    split_col1, split_col2 = st.columns(2)

    with split_col1:
        st.markdown("#### Distribuicao das Parciais")
        split_counts, by_cat = compute_negative_positive_split(filtered_results, selected_categories)
        if not split_counts.empty:
            split_colors = {
                "Positivo": COLOR_PALETTE["accent"],
                "Negativo": COLOR_PALETTE["success"],
                "Uniforme": COLOR_PALETTE["primary"],
                "Positive": COLOR_PALETTE["accent"],
                "Negative": COLOR_PALETTE["success"],
                "Even": COLOR_PALETTE["primary"],
            }
            fig_pie = go.Figure()
            fig_pie.add_trace(
                go.Pie(
                    labels=split_counts["split_type"],
                    values=split_counts["count"],
                    marker_colors=[split_colors.get(st, COLOR_PALETTE["muted"]) for st in split_counts["split_type"]],
                    hole=0.45,
                    textinfo="label+percent",
                    textposition="outside",
                )
            )
            fig_pie = apply_standard_layout(fig_pie, title="Proporcao de Parciais Negativas vs Positivas")
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados de parciais disponiveis"), use_container_width=True)

    with split_col2:
        st.markdown("#### % Parciais Negativas por Categoria")
        if not by_cat.empty:
            fig_bar_split = go.Figure()
            cat_order = [c for c in ["Elite", "Advanced", "Intermediate", "Recreational"] if c in by_cat["performance_category"].values]
            by_cat_sorted = by_cat.set_index("performance_category").reindex(cat_order).reset_index()
            fig_bar_split.add_trace(
                go.Bar(
                    x=by_cat_sorted["performance_category"],
                    y=by_cat_sorted["pct_negative"],
                    marker_color=[PERFORMANCE_COLORS.get(c, COLOR_PALETTE["muted"]) for c in by_cat_sorted["performance_category"]],
                    text=by_cat_sorted["pct_negative"].apply(lambda v: f"{v:.1f}%"),
                    textposition="outside",
                )
            )
            fig_bar_split.update_layout(
                xaxis_title="Categoria de Desempenho",
                yaxis_title="% Parciais Negativas",
                yaxis_tickformat=".1f",
            )
            fig_bar_split = apply_standard_layout(fig_bar_split, title="Taxa de Parciais Negativas por Categoria")
            st.plotly_chart(fig_bar_split, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados de parciais disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Mapa de Calor de Ritmo por Segmento")
    heatmap_pivot = compute_segment_heatmap(filtered_splits, selected_races)
    if not heatmap_pivot.empty:
        fig_heatmap = go.Figure()
        fig_heatmap.add_trace(
            go.Heatmap(
                z=heatmap_pivot.values,
                x=heatmap_pivot.columns.tolist(),
                y=heatmap_pivot.index.tolist(),
                colorscale="YlOrRd",
                text=[[f"{val:.2f}" if not pd.isna(val) else "" for val in row] for row in heatmap_pivot.values],
                texttemplate="%{text}",
                textfont={"size": 10},
                colorbar=dict(title="min/km"),
            )
        )
        fig_heatmap.update_layout(
            xaxis_title="Segmento",
            yaxis_title="Corrida",
        )
        fig_heatmap = apply_standard_layout(fig_heatmap, title="Ritmo Medio por Segmento e Corrida (min/km)")
        st.plotly_chart(fig_heatmap, use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("Sem dados de ritmo por segmento disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Queda de Ritmo Apos 30 km")
    dropoff_data = compute_pace_dropoff(filtered_results, selected_races)
    if not dropoff_data.empty:
        fig_dropoff = go.Figure()
        fig_dropoff.add_trace(
            go.Bar(
                x=dropoff_data["race"],
                y=dropoff_data["pace_increase_pct"],
                marker_color=[
                    COLOR_PALETTE["accent"] if v > 10 else COLOR_PALETTE["warning"] if v > 5 else COLOR_PALETTE["success"]
                    for v in dropoff_data["pace_increase_pct"]
                ],
                text=dropoff_data["pace_increase_pct"].apply(lambda v: f"{v:.1f}%"),
                textposition="outside",
            )
        )
        fig_dropoff.update_layout(
            xaxis_title="Corrida",
            yaxis_title="Aumento do Ritmo (%)",
        )
        fig_dropoff = apply_standard_layout(fig_dropoff, title="Aumento do Ritmo da Primeira para a Segunda Metade por Corrida")
        st.plotly_chart(fig_dropoff, use_container_width=True)
        st.markdown(
            "Este grafico mostra o percentual de aumento no ritmo medio da primeira para a segunda "
            "metade da maratona em cada corrida. Valores mais altos indicam uma maior desaceleracao "
            "na segunda metade, onde o 'muro' da maratona impacta mais significativamente o desempenho. "
            "Corridas com terreno ou clima mais desafiadores tendem a apresentar quedas maiores."
        )
    else:
        st.plotly_chart(create_empty_figure("Sem dados de queda de ritmo disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Tabela de Estatisticas das Parciais")
    stats_table = compute_segment_stats_table(filtered_splits, selected_categories)
    if not stats_table.empty:
        display_table = stats_table.copy()
        for col in display_table.columns:
            if col != "Categoria":
                display_table[col] = display_table[col].apply(
                    lambda v: f"{v:.2f}" if pd.notna(v) else "N/A"
                )
        st.dataframe(display_table, use_container_width=True, hide_index=True)
    else:
        st.info("Sem estatisticas de parciais disponiveis para os filtros selecionados.")

    st.divider()

    with st.expander("Analise de Ritmo Uniforme"):
        st.markdown("#### Corredores Mais Regulares")
        st.markdown(
            "Um ritmo uniforme e a marca de uma estrategia de maratona bem executada. Corredores "
            "que mantem um ritmo constante ao longo da prova geralmente alcancam um desempenho "
            "geral melhor e sao menos propensos a experimentar uma desaceleracao dramatica nos "
            "quilometros finais. A tabela abaixo identifica os corredores com a menor variacao "
            "de ritmo em todos os segmentos de 5 km."
        )
        even_pace_runners = compute_even_pace_runners(filtered_splits)
        if not even_pace_runners.empty:
            display_runners = even_pace_runners[["runner_name", "marathon", "year"]].copy()
            display_runners.columns = ["Corredor", "Corrida", "Ano"]
            if "pace_std" in even_pace_runners.columns:
                display_runners["Desvio Padrao do Ritmo (min/km)"] = even_pace_runners["pace_std"].apply(
                    lambda v: f"{v / 60:.3f}" if pd.notna(v) else "N/A"
                )
            if "finish_time" in even_pace_runners.columns:
                display_runners["Tempo Final"] = even_pace_runners["finish_time"]
            elif "finish_time_sec" in even_pace_runners.columns:
                display_runners["Tempo Final"] = even_pace_runners["finish_time_sec"].apply(
                    lambda s: seconds_to_time(s) if pd.notna(s) and s > 0 else "N/A"
                )
            st.dataframe(display_runners, use_container_width=True, hide_index=True)
            st.markdown(
                "Um desvio padrao de ritmo menor indica uma distribuicao de ritmo mais uniforme. "
                "Pesquisas mostram que um ritmo regular esta fortemente correlacionado com um "
                "melhor desempenho na maratona, pois otimiza a distribuicao de energia e minimiza "
                "o risco de deplecao de glicogenio."
            )
        else:
            st.info("Sem dados de variacao de ritmo disponiveis para os filtros selecionados.")


if __name__ == "__main__" or "streamlit" in os.path.basename(sys.argv[0]).lower():
    data_loaded = False
    results_df = None
    pace_splits_df = None

    if "marathon_results" in st.session_state and st.session_state["marathon_results"] is not None:
        results_df = st.session_state["marathon_results"]
        data_loaded = True
    elif "results" in st.session_state and st.session_state["results"] is not None:
        results_df = st.session_state["results"]
        data_loaded = True
    elif "combined_data" in st.session_state and st.session_state["combined_data"] is not None:
        results_df = st.session_state["combined_data"]
        data_loaded = True

    if "pace_splits" in st.session_state and st.session_state["pace_splits"] is not None:
        pace_splits_df = st.session_state["pace_splits"]
    elif "pace_splits_data" in st.session_state and st.session_state["pace_splits_data"] is not None:
        pace_splits_df = st.session_state["pace_splits_data"]

    if not data_loaded:
        try:
            from src.data.load_data import load_csv, get_data_filepaths

            raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
            processed_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

            results_path = os.path.normpath(os.path.join(raw_dir, "marathon_results.csv"))
            pace_splits_path = os.path.normpath(os.path.join(processed_dir, "pace_splits_analysis.csv"))

            if os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                data_loaded = True

            if os.path.exists(pace_splits_path):
                pace_splits_df = pd.read_csv(pace_splits_path)
        except Exception:
            pass

    if data_loaded or pace_splits_df is not None:
        render_pace_splits_page(results_df, pace_splits_df)
    else:
        st.warning(
            "Nenhum dado carregado. Certifique-se de que os conjuntos de dados de maratona estao "
            "disponiveis no diretorio de dados ou carregue-os atraves do ponto de entrada principal "
            "do aplicativo para que sejam armazenados no estado da sessao."
        )
