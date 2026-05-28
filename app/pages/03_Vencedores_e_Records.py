import streamlit as st
import pandas as pd
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
    plot_winner_time_evolution,
    plot_winners_by_country,
    create_empty_figure,
    apply_standard_layout,
    COLOR_PALETTE,
    RACE_COLORS,
)

from src.utils.helpers import (
    seconds_to_time,
    calculate_pace_per_km,
    format_pace_str,
)

from src.utils.config import (
    MARATHON_DISTANCE_KM,
    RACE_NAMES,
    YEARS,
    RAW_DIR,
)


@st.cache_data
def load_winners_data():
    path = os.path.join(RAW_DIR, "winners_data.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def load_results_data():
    raw_path = os.path.join(RAW_DIR, "marathon_results.csv")
    if os.path.exists(raw_path):
        return pd.read_csv(raw_path)
    return None


@st.cache_data
def load_metadata_data():
    path = os.path.join(RAW_DIR, "race_metadata.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def build_champions_timeline(winners_df, selected_years, selected_races, selected_gender):
    if winners_df is None or winners_df.empty:
        return pd.DataFrame()
    working = winners_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    if selected_years:
        working = working[working["year"].isin(selected_years)]
    if selected_races:
        working = working[working[race_col].isin(selected_races)]
    if selected_gender != "Todos" and "gender" in working.columns:
        gender_val = "M" if selected_gender == "Masculino" else "F"
        working = working[working["gender"] == gender_val]
    time_col = "winning_time_sec" if "winning_time_sec" in working.columns else "finish_time_sec"
    country_col = "winner_country" if "winner_country" in working.columns else "country"
    name_col = "winner_name" if "winner_name" in working.columns else "runner_name"
    male = working[working["gender"] == "M"].sort_values(["year", race_col])
    female = working[working["gender"] == "F"].sort_values(["year", race_col])
    male_display = male[["year", race_col, name_col, time_col, country_col]].copy()
    male_display.columns = ["Ano", "Corrida", "Campeao (M)", "Tempo Seg (M)", "Pais (M)"]
    male_display["Tempo (M)"] = male_display["Tempo Seg (M)"].apply(
        lambda x: seconds_to_time(x) if pd.notna(x) and x > 0 else ""
    )
    female_display = female[["year", race_col, name_col, time_col, country_col]].copy()
    female_display.columns = ["Ano", "Corrida", "Campea (F)", "Tempo Seg (F)", "Pais (F)"]
    female_display["Tempo (F)"] = female_display["Tempo Seg (F)"].apply(
        lambda x: seconds_to_time(x) if pd.notna(x) and x > 0 else ""
    )
    male_display = male_display.drop(columns=["Tempo Seg (M)"])
    female_display = female_display.drop(columns=["Tempo Seg (F)"])
    timeline = pd.merge(
        male_display,
        female_display,
        on=["Ano", "Corrida"],
        how="outer",
    ).sort_values(["Ano", "Corrida"]).reset_index(drop=True)
    return timeline


@st.cache_data
def build_pace_of_winners_data(winners_df, selected_years, selected_races, selected_gender):
    if winners_df is None or winners_df.empty:
        return pd.DataFrame()
    working = winners_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    time_col = "winning_time_sec" if "winning_time_sec" in working.columns else "finish_time_sec"
    if selected_years:
        working = working[working["year"].isin(selected_years)]
    if selected_races:
        working = working[working[race_col].isin(selected_races)]
    if selected_gender != "Todos" and "gender" in working.columns:
        gender_val = "M" if selected_gender == "Masculino" else "F"
        working = working[working["gender"] == gender_val]
    working = working[working[time_col].notna() & (working[time_col] > 0)].copy()
    working["pace_per_km"] = working[time_col].apply(calculate_pace_per_km)
    working["pace_str"] = working["pace_per_km"].apply(format_pace_str)
    return working


@st.cache_data
def build_country_wins_data(winners_df, selected_years, selected_races, selected_gender):
    if winners_df is None or winners_df.empty:
        return pd.DataFrame(), pd.DataFrame()
    working = winners_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    country_col = "winner_country" if "winner_country" in working.columns else "country"
    if selected_years:
        working = working[working["year"].isin(selected_years)]
    if selected_races:
        working = working[working[race_col].isin(selected_races)]
    if selected_gender != "Todos" and "gender" in working.columns:
        gender_val = "M" if selected_gender == "Masculino" else "F"
        working = working[working["gender"] == gender_val]
    counts = working[country_col].value_counts().reset_index()
    counts.columns = ["country", "wins"]
    bar_data = counts.copy()
    pie_data = counts.copy()
    return bar_data, pie_data


@st.cache_data
def build_course_records_table(winners_df, metadata_df):
    records = []
    races = ["Tokyo", "Boston", "London", "Berlin", "Chicago", "New York City"]
    if winners_df is not None and not winners_df.empty:
        race_col = "marathon" if "marathon" in winners_df.columns else "race"
        time_col = "winning_time_sec" if "winning_time_sec" in winners_df.columns else "finish_time_sec"
        name_col = "winner_name" if "winner_name" in winners_df.columns else "runner_name"
        country_col = "winner_country" if "winner_country" in winners_df.columns else "country"
        for race in races:
            race_data = winners_df[winners_df[race_col] == race]
            if race_data.empty:
                continue
            male_data = race_data[race_data["gender"] == "M"]
            female_data = race_data[race_data["gender"] == "F"]
            if not male_data.empty and time_col in male_data.columns:
                best_male_idx = male_data[time_col].idxmin()
                best_male_row = male_data.loc[best_male_idx]
                male_record_sec = best_male_row[time_col]
                male_record_time = seconds_to_time(male_record_sec)
                male_record_name = best_male_row[name_col] if name_col in best_male_row.index else "N/A"
                male_record_country = best_male_row[country_col] if country_col in best_male_row.index else "N/A"
                male_record_year = int(best_male_row["year"]) if "year" in best_male_row.index else "N/A"
            else:
                male_record_time = "N/A"
                male_record_name = "N/A"
                male_record_country = "N/A"
                male_record_year = "N/A"
                male_record_sec = None
            if not female_data.empty and time_col in female_data.columns:
                best_female_idx = female_data[time_col].idxmin()
                best_female_row = female_data.loc[best_female_idx]
                female_record_sec = best_female_row[time_col]
                female_record_time = seconds_to_time(female_record_sec)
                female_record_name = best_female_row[name_col] if name_col in best_female_row.index else "N/A"
                female_record_country = best_female_row[country_col] if country_col in best_female_row.index else "N/A"
                female_record_year = int(best_female_row["year"]) if "year" in best_female_row.index else "N/A"
            else:
                female_record_time = "N/A"
                female_record_name = "N/A"
                female_record_country = "N/A"
                female_record_year = "N/A"
                female_record_sec = None
            records.append({
                "Corrida": race,
                "Recorde Masculino": male_record_time,
                "Atleta (M)": male_record_name,
                "Pais (M)": male_record_country,
                "Ano (M)": male_record_year,
                "Recorde Masculino Seg": male_record_sec,
                "Recorde Feminino": female_record_time,
                "Atleta (F)": female_record_name,
                "Pais (F)": female_record_country,
                "Ano (F)": female_record_year,
                "Recorde Feminino Seg": female_record_sec,
            })
    if metadata_df is not None and not metadata_df.empty:
        meta_race_col = "marathon" if "marathon" in metadata_df.columns else "race"
        for record in records:
            race_name = record["Corrida"]
            meta_rows = metadata_df[metadata_df[meta_race_col] == race_name]
            if not meta_rows.empty:
                latest_meta = meta_rows.sort_values("year", ascending=False).iloc[0]
                if "course_record_male_time" in latest_meta.index and record["Recorde Masculino"] == "N/A":
                    record["Recorde Masculino"] = latest_meta["course_record_male_time"]
                    if "course_record_male_name" in latest_meta.index:
                        record["Atleta (M)"] = latest_meta["course_record_male_name"]
                    if "course_record_male_year" in latest_meta.index:
                        record["Ano (M)"] = int(latest_meta["course_record_male_year"])
                if "course_record_female_time" in latest_meta.index and record["Recorde Feminino"] == "N/A":
                    record["Recorde Feminino"] = latest_meta["course_record_female_time"]
                    if "course_record_female_name" in latest_meta.index:
                        record["Atleta (F)"] = latest_meta["course_record_female_name"]
                    if "course_record_female_year" in latest_meta.index:
                        record["Ano (F)"] = int(latest_meta["course_record_female_year"])
    records_df = pd.DataFrame(records)
    return records_df


@st.cache_data
def build_dominant_athletes_table(winners_df):
    if winners_df is None or winners_df.empty:
        return pd.DataFrame()
    working = winners_df.copy()
    name_col = "winner_name" if "winner_name" in working.columns else "runner_name"
    country_col = "winner_country" if "winner_country" in working.columns else "country"
    time_col = "winning_time_sec" if "winning_time_sec" in working.columns else "finish_time_sec"
    race_col = "marathon" if "marathon" in working.columns else "race"
    athlete_stats = working.groupby(name_col).agg(
        total_wins=(name_col, "size"),
        country=(country_col, "first"),
        best_time_sec=(time_col, "min"),
        first_year=("year", "min"),
        last_year=("year", "max"),
        races_won=(race_col, lambda x: ", ".join(sorted(x.unique()))),
    ).reset_index()
    athlete_stats.columns = [
        "Atleta", "Vitorias WMM", "Pais",
        "Melhor Tempo Seg", "Primeira Vitoria", "Ultima Vitoria", "Corridas Vencidas",
    ]
    athlete_stats["Melhor Tempo"] = athlete_stats["Melhor Tempo Seg"].apply(
        lambda x: seconds_to_time(x) if pd.notna(x) and x > 0 else "N/A"
    )
    athlete_stats["Periodo"] = (
        athlete_stats["Primeira Vitoria"].astype(str)
        + " - "
        + athlete_stats["Ultima Vitoria"].astype(str)
    )
    athlete_stats = athlete_stats.sort_values("Vitorias WMM", ascending=False).reset_index(drop=True)
    top_athletes = athlete_stats[
        ["Atleta", "Pais", "Vitorias WMM", "Periodo", "Melhor Tempo", "Corridas Vencidas"]
    ].head(15)
    return top_athletes


@st.cache_data
def build_winning_time_stats(winners_df):
    if winners_df is None or winners_df.empty:
        return {}
    working = winners_df.copy()
    time_col = "winning_time_sec" if "winning_time_sec" in working.columns else "finish_time_sec"
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    stats = {}
    all_times = working[time_col]
    stats["all_avg"] = all_times.mean()
    stats["all_fastest"] = all_times.min()
    stats["all_slowest"] = all_times.max()
    if "gender" in working.columns:
        male_times = working[working["gender"] == "M"][time_col]
        female_times = working[working["gender"] == "F"][time_col]
        if len(male_times) > 0:
            stats["male_avg"] = male_times.mean()
            stats["male_fastest"] = male_times.min()
            stats["male_slowest"] = male_times.max()
        else:
            stats["male_avg"] = 0
            stats["male_fastest"] = 0
            stats["male_slowest"] = 0
        if len(female_times) > 0:
            stats["female_avg"] = female_times.mean()
            stats["female_fastest"] = female_times.min()
            stats["female_slowest"] = female_times.max()
        else:
            stats["female_avg"] = 0
            stats["female_fastest"] = 0
            stats["female_slowest"] = 0
    return stats


def render_winners_page(winners_df, results_df, metadata_df):
    st.title("Vencedores e Recordes")
    st.markdown(
        "Acompanhe os maratonistas mais rapidos em todas as seis Abbott World Marathon Majors "
        "de 2018 a 2025. Explore as tendencias dos tempos vencedores, os recordes dos percursos, "
        "as nacoes dominantes e os atletas mais condecorados na historia das maratonas de elite."
    )
    st.divider()

    race_col = "marathon" if winners_df is not None and "marathon" in winners_df.columns else "race"
    if winners_df is not None and not winners_df.empty:
        available_years = sorted(winners_df["year"].dropna().unique().tolist())
        available_races = sorted(winners_df[race_col].dropna().unique().tolist())
    else:
        available_years = list(range(2018, 2026))
        available_races = ["Berlin", "Boston", "Chicago", "London", "New York City", "Tokyo"]

    with st.expander("Filtros", expanded=False):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            selected_years = st.multiselect(
                "Ano",
                options=available_years,
                default=available_years,
                key="winners_year_filter",
            )
        with filter_col2:
            selected_races = st.multiselect(
                "Corrida",
                options=available_races,
                default=available_races,
                key="winners_race_filter",
            )
        with filter_col3:
            selected_gender = st.radio(
                "Genero",
                options=["Todos", "Masculino", "Feminino"],
                index=0,
                key="winners_gender_filter",
            )

    st.divider()

    st.subheader("Linha do Tempo dos Campeoes")
    st.markdown(
        "Listagem completa dos campeoes masculinos e femininos de cada corrida "
        "World Marathon Major por ano."
    )
    timeline_df = build_champions_timeline(winners_df, selected_years, selected_races, selected_gender)
    if not timeline_df.empty:
        display_cols = [
            "Ano", "Corrida", "Campeao (M)", "Tempo (M)", "Pais (M)",
            "Campea (F)", "Tempo (F)", "Pais (F)",
        ]
        existing_cols = [c for c in display_cols if c in timeline_df.columns]
        st.dataframe(
            timeline_df[existing_cols],
            use_container_width=True,
            height=400,
            hide_index=True,
        )
    else:
        st.info("Nenhum dado de campeoes disponivel para os filtros selecionados.")

    st.divider()

    st.subheader("Evolucao dos Tempos dos Vencedores")
    st.markdown(
        "Acompanhe como os tempos vencedores mudaram ao longo dos anos para cada maratona, "
        "separados por genero."
    )
    evo_col1, evo_col2 = st.columns(2)

    with evo_col1:
        st.markdown("#### Tempos Masculinos")
        if winners_df is not None and not winners_df.empty:
            fig_male = plot_winner_time_evolution(winners_df, gender="M")
            st.plotly_chart(fig_male, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados de vencedores disponiveis"), use_container_width=True)

    with evo_col2:
        st.markdown("#### Tempos Femininos")
        if winners_df is not None and not winners_df.empty:
            fig_female = plot_winner_time_evolution(winners_df, gender="F")
            st.plotly_chart(fig_female, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados de vencedores disponiveis"), use_container_width=True)

    st.divider()

    st.subheader("Ritmo dos Vencedores")
    st.markdown(
        "Ritmo vencedor por quilometro por corrida e ano, com o ritmo mais rapido destacado."
    )
    pace_data = build_pace_of_winners_data(winners_df, selected_years, selected_races, selected_gender)
    if not pace_data.empty:
        fastest_pace = pace_data["pace_per_km"].min()
        fig_pace = go.Figure()
        race_col_pace = "marathon" if "marathon" in pace_data.columns else "race"
        for race in sorted(pace_data[race_col_pace].unique()):
            race_subset = pace_data[pace_data[race_col_pace] == race].sort_values("year")
            fig_pace.add_trace(
                go.Bar(
                    x=race_subset["year"].astype(str),
                    y=race_subset["pace_per_km"],
                    name=race,
                    marker_color=RACE_COLORS.get(race, COLOR_PALETTE["primary"]),
                    text=race_subset["pace_str"],
                    textposition="outside",
                    textfont=dict(size=9),
                )
            )
        fig_pace.add_hline(
            y=fastest_pace,
            line_dash="dash",
            line_color=COLOR_PALETTE["success"],
            annotation_text=f"Mais rapido: {format_pace_str(fastest_pace)} /km",
            annotation_position="top left",
        )
        fig_pace.update_layout(
            xaxis_title="Ano",
            yaxis_title="Ritmo (min/km)",
            barmode="group",
            legend_title="Corrida",
        )
        fig_pace = apply_standard_layout(fig_pace, title="Ritmo Vencedor por km por Corrida e Ano")
        st.plotly_chart(fig_pace, use_container_width=True)
    else:
        st.info("Nenhum dado de ritmo disponivel para os filtros selecionados.")

    st.divider()

    st.subheader("Paises com Mais Vitorias")
    st.markdown(
        "Distribuicao das vitorias por pais de origem em todas as World Marathon Majors."
    )
    country_col1, country_col2 = st.columns(2)

    bar_data, pie_data = build_country_wins_data(winners_df, selected_years, selected_races, selected_gender)

    with country_col1:
        st.markdown("#### Vitorias por Pais")
        if not bar_data.empty:
            fig_country_bar = go.Figure()
            fig_country_bar.add_trace(
                go.Bar(
                    x=bar_data["country"],
                    y=bar_data["wins"],
                    marker_color=COLOR_PALETTE["accent"],
                    text=bar_data["wins"],
                    textposition="outside",
                )
            )
            fig_country_bar.update_layout(
                xaxis_title="Pais",
                yaxis_title="Numero de Vitorias",
            )
            fig_country_bar = apply_standard_layout(fig_country_bar, title="Vitorias por Pais")
            st.plotly_chart(fig_country_bar, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados de paises disponiveis"), use_container_width=True)

    with country_col2:
        st.markdown("#### Distribuicao das Vitorias")
        if not pie_data.empty:
            fig_pie = go.Figure()
            fig_pie.add_trace(
                go.Pie(
                    labels=pie_data["country"],
                    values=pie_data["wins"],
                    hole=0.45,
                    textinfo="label+percent",
                    textposition="outside",
                    marker_colors=px.colors.qualitative.Set2[: len(pie_data)],
                )
            )
            fig_pie.update_layout(
                title=dict(
                    text="Participacao nas Vitorias por Pais",
                    font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                    x=0.5,
                    xanchor="center",
                ),
                template="plotly_white",
                paper_bgcolor=COLOR_PALETTE["card_bg"],
                font=dict(family="Arial", size=12, color=COLOR_PALETTE["text"]),
                margin=dict(l=30, r=30, t=60, b=30),
                showlegend=True,
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados de paises disponiveis"), use_container_width=True)

    st.divider()

    with st.expander("Recordes dos Percursos", expanded=False):
        st.subheader("Recordes dos Percursos")
        st.markdown(
            "Recordes dos percursos de cada World Marathon Major, baseados no conjunto de dados "
            "disponivel (2018-2025) e complementados com dados historicos de recordes quando disponiveis."
        )
        records_df = build_course_records_table(winners_df, metadata_df)
        if not records_df.empty:
            display_records = records_df[
                [
                    "Corrida", "Recorde Masculino", "Atleta (M)", "Pais (M)", "Ano (M)",
                    "Recorde Feminino", "Atleta (F)", "Pais (F)", "Ano (F)",
                ]
            ].copy()
            if "Recorde Masculino Seg" in records_df.columns and records_df["Recorde Masculino Seg"].notna().any():
                fastest_male_sec = records_df["Recorde Masculino Seg"].min()
                fastest_male_race = records_df.loc[
                    records_df["Recorde Masculino Seg"] == fastest_male_sec, "Corrida"
                ].values[0]
                st.markdown(
                    f"**Recorde Masculino mais rapido:** {seconds_to_time(fastest_male_sec)} ({fastest_male_race})"
                )
            if "Recorde Feminino Seg" in records_df.columns and records_df["Recorde Feminino Seg"].notna().any():
                fastest_female_sec = records_df["Recorde Feminino Seg"].min()
                fastest_female_race = records_df.loc[
                    records_df["Recorde Feminino Seg"] == fastest_female_sec, "Corrida"
                ].values[0]
                st.markdown(
                    f"**Recorde Feminino mais rapido:** {seconds_to_time(fastest_female_sec)} ({fastest_female_race})"
                )
            st.dataframe(
                display_records,
                use_container_width=True,
                hide_index=True,
            )
            st.markdown(
                "A **Maratona de Berlim** e amplamente reconhecida como o percurso de maratona mais rapido "
                "do mundo, tendo sediado diversas performances de recordes mundiais. Seu trajeto plano e "
                "rapido pela capital alema oferece condicoes ideais para corredores de elite em busca de "
                "melhores tempos pessoais e mundiais."
            )
        else:
            st.info("Nenhum dado de recordes de percursos disponivel.")

    st.divider()

    st.subheader("Atletas Mais Dominantes")
    st.markdown(
        "Atletas com o maior numero de vitorias nas World Marathon Majors no conjunto de dados, "
        "destacando a excelencia sustentada ao longo de multiplas corridas e anos."
    )
    dominant_df = build_dominant_athletes_table(winners_df)
    if not dominant_df.empty:
        highlight_mask = dominant_df["Atleta"] == "Eliud Kipchoge"
        if highlight_mask.any():
            kipchoge_wins = dominant_df.loc[highlight_mask, "Vitorias WMM"].values[0]
            st.success(
                f"Eliud Kipchoge lidera com {int(kipchoge_wins)} vitorias WMM de 2018 a 2025."
            )
        st.dataframe(
            dominant_df,
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Nenhum dado de atletas disponivel.")

    st.divider()

    st.subheader("Estatisticas dos Tempos Vencedores")
    st.markdown(
        "Estatisticas resumidas dos tempos vencedores em todas as World Marathon Majors, "
        "separadas por genero."
    )
    winning_stats = build_winning_time_stats(winners_df)
    if winning_stats:
        stat_col1, stat_col2, stat_col3 = st.columns(3)
        with stat_col1:
            st.markdown("#### Geral")
            if "all_avg" in winning_stats and winning_stats["all_avg"] > 0:
                st.metric("Tempo Medio Vencedor", seconds_to_time(winning_stats["all_avg"]))
            else:
                st.metric("Tempo Medio Vencedor", "N/A")
            if "all_fastest" in winning_stats and winning_stats["all_fastest"] > 0:
                st.metric("Tempo Mais Rapido", seconds_to_time(winning_stats["all_fastest"]))
            else:
                st.metric("Tempo Mais Rapido", "N/A")
            if "all_slowest" in winning_stats and winning_stats["all_slowest"] > 0:
                st.metric("Tempo Mais Lento", seconds_to_time(winning_stats["all_slowest"]))
            else:
                st.metric("Tempo Mais Lento", "N/A")
        with stat_col2:
            st.markdown("#### Masculino")
            if "male_avg" in winning_stats and winning_stats["male_avg"] > 0:
                st.metric("Tempo Medio Vencedor", seconds_to_time(winning_stats["male_avg"]))
            else:
                st.metric("Tempo Medio Vencedor", "N/A")
            if "male_fastest" in winning_stats and winning_stats["male_fastest"] > 0:
                st.metric("Tempo Mais Rapido", seconds_to_time(winning_stats["male_fastest"]))
            else:
                st.metric("Tempo Mais Rapido", "N/A")
            if "male_slowest" in winning_stats and winning_stats["male_slowest"] > 0:
                st.metric("Tempo Mais Lento", seconds_to_time(winning_stats["male_slowest"]))
            else:
                st.metric("Tempo Mais Lento", "N/A")
        with stat_col3:
            st.markdown("#### Feminino")
            if "female_avg" in winning_stats and winning_stats["female_avg"] > 0:
                st.metric("Tempo Medio Vencedor", seconds_to_time(winning_stats["female_avg"]))
            else:
                st.metric("Tempo Medio Vencedor", "N/A")
            if "female_fastest" in winning_stats and winning_stats["female_fastest"] > 0:
                st.metric("Tempo Mais Rapido", seconds_to_time(winning_stats["female_fastest"]))
            else:
                st.metric("Tempo Mais Rapido", "N/A")
            if "female_slowest" in winning_stats and winning_stats["female_slowest"] > 0:
                st.metric("Tempo Mais Lento", seconds_to_time(winning_stats["female_slowest"]))
            else:
                st.metric("Tempo Mais Lento", "N/A")
    else:
        st.info("Nenhuma estatistica de tempos vencedores disponivel.")


if __name__ == "__main__" or "streamlit" in os.path.basename(sys.argv[0]).lower():
    winners_df = None
    results_df = None
    metadata_df = None

    if "winners" in st.session_state and st.session_state["winners"] is not None:
        winners_df = st.session_state["winners"]
    elif "winners_data" in st.session_state and st.session_state["winners_data"] is not None:
        winners_df = st.session_state["winners_data"]

    if "marathon_results" in st.session_state and st.session_state["marathon_results"] is not None:
        results_df = st.session_state["marathon_results"]
    elif "results" in st.session_state and st.session_state["results"] is not None:
        results_df = st.session_state["results"]
    elif "combined_data" in st.session_state and st.session_state["combined_data"] is not None:
        results_df = st.session_state["combined_data"]

    if "metadata" in st.session_state and st.session_state["metadata"] is not None:
        metadata_df = st.session_state["metadata"]
    elif "race_metadata" in st.session_state and st.session_state["race_metadata"] is not None:
        metadata_df = st.session_state["race_metadata"]

    if winners_df is None:
        winners_df = load_winners_data()

    if results_df is None:
        results_df = load_results_data()

    if metadata_df is None:
        metadata_df = load_metadata_data()

    if winners_df is not None:
        render_winners_page(winners_df, results_df, metadata_df)
    else:
        st.warning(
            "Nenhum dado carregado. Certifique-se de que os conjuntos de dados de maratona estao "
            "disponiveis no diretorio de dados, ou carregue-os pelo ponto de entrada principal do "
            "aplicativo para que sejam armazenados no estado da sessao."
        )
