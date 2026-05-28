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
    plot_boxplot_times_by_race,
    plot_covid_impact,
    plot_gender_comparison,
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
    RACE_NAMES,
    RACE_CITIES,
    RACE_COUNTRIES,
    YEARS,
    MARATHON_DISTANCE_KM,
)


@st.cache_data
def compute_race_summary_table(results_df, years, genders, races):
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    working = results_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    if "year" in working.columns and years:
        working = working[working["year"].isin(years)]
    if "gender" in working.columns and genders:
        working = working[working["gender"].isin(genders)]
    if race_col in working.columns and races:
        working = working[working[race_col].isin(races)]
    if "status" in working.columns:
        working = working[working["status"] == "Finished"]
    if time_col not in working.columns or race_col not in working.columns:
        return pd.DataFrame()
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    rows = []
    for race_name in sorted(working[race_col].unique()):
        race_data = working[working[race_col] == race_name]
        if race_data.empty:
            continue
        cidade = ""
        pais = ""
        if "city" in race_data.columns:
            cidade = race_data["city"].mode().iloc[0] if len(race_data["city"].mode()) > 0 else ""
        elif race_name in RACE_NAMES:
            idx = RACE_NAMES.index(race_name)
            cidade = RACE_CITIES[idx]
        if "country" in race_data.columns:
            pais = race_data["country"].mode().iloc[0] if len(race_data["country"].mode()) > 0 else ""
        elif race_name in RACE_NAMES:
            idx = RACE_NAMES.index(race_name)
            pais = RACE_COUNTRIES[idx]
        total_finishers = len(race_data)
        avg_finish_time = race_data[time_col].mean()
        fastest_time = race_data[time_col].min()
        avg_pace = calculate_pace_per_km(avg_finish_time)
        rows.append({
            "Corrida": race_name,
            "Cidade": cidade,
            "Pais": pais,
            "Total de Finalistas": total_finishers,
            "Tempo Medio": seconds_to_time(avg_finish_time),
            "Tempo Mais Rapido": seconds_to_time(fastest_time),
            "Ritmo Medio": format_pace_str(avg_pace) + " /km",
        })
    return pd.DataFrame(rows)


@st.cache_data
def compute_fastest_marathons_ranking(results_df, years, genders, races):
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    working = results_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    if "year" in working.columns and years:
        working = working[working["year"].isin(years)]
    if "gender" in working.columns and genders:
        working = working[working["gender"].isin(genders)]
    if race_col in working.columns and races:
        working = working[working[race_col].isin(races)]
    if "status" in working.columns:
        working = working[working["status"] == "Finished"]
    if time_col not in working.columns or race_col not in working.columns:
        return pd.DataFrame()
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    avg_by_race = working.groupby(race_col)[time_col].mean().reset_index()
    avg_by_race.columns = ["Corrida", "Tempo Medio (seg)"]
    avg_by_race["Tempo Medio (horas)"] = avg_by_race["Tempo Medio (seg)"] / 3600
    avg_by_race["Tempo Medio"] = avg_by_race["Tempo Medio (seg)"].apply(seconds_to_time)
    avg_by_race = avg_by_race.sort_values("Tempo Medio (seg)").reset_index(drop=True)
    avg_by_race["Posicao"] = range(1, len(avg_by_race) + 1)
    return avg_by_race


@st.cache_data
def compute_pace_statistics(results_df, years, genders, races):
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    working = results_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    if "year" in working.columns and years:
        working = working[working["year"].isin(years)]
    if "gender" in working.columns and genders:
        working = working[working["gender"].isin(genders)]
    if race_col in working.columns and races:
        working = working[working[race_col].isin(races)]
    if "status" in working.columns:
        working = working[working["status"] == "Finished"]
    if time_col not in working.columns or race_col not in working.columns:
        return pd.DataFrame()
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    rows = []
    for race_name in sorted(working[race_col].unique()):
        race_data = working[working[race_col] == race_name]
        if race_data.empty:
            continue
        paces = race_data[time_col].apply(lambda t: calculate_pace_per_km(t))
        avg_pace = paces.mean()
        median_pace = paces.median()
        min_pace = paces.min()
        max_pace = paces.max()
        std_pace = paces.std()
        rows.append({
            "Corrida": race_name,
            "Ritmo Medio (/km)": format_pace_str(avg_pace),
            "Ritmo Mediano (/km)": format_pace_str(median_pace),
            "Ritmo Mais Rapido (/km)": format_pace_str(min_pace),
            "Ritmo Mais Lento (/km)": format_pace_str(max_pace),
            "Desvio Padrao (/km)": format_pace_str(std_pace),
        })
    return pd.DataFrame(rows)


@st.cache_data
def compute_participant_volume(results_df, years, genders, races):
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    working = results_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    if "year" in working.columns and years:
        working = working[working["year"].isin(years)]
    if "gender" in working.columns and genders:
        working = working[working["gender"].isin(genders)]
    if race_col in working.columns and races:
        working = working[working[race_col].isin(races)]
    if "status" in working.columns:
        working = working[working["status"] == "Finished"]
    if "year" not in working.columns or race_col not in working.columns:
        return pd.DataFrame()
    counts = working.groupby(["year", race_col]).size().reset_index(name="Finalistas")
    return counts


@st.cache_data
def compute_avg_pace_by_race(results_df, years, genders, races):
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    working = results_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    time_col = "finish_time_sec" if "finish_time_sec" in working.columns else "finish_seconds"
    if "year" in working.columns and years:
        working = working[working["year"].isin(years)]
    if "gender" in working.columns and genders:
        working = working[working["gender"].isin(genders)]
    if race_col in working.columns and races:
        working = working[working[race_col].isin(races)]
    if "status" in working.columns:
        working = working[working["status"] == "Finished"]
    if time_col not in working.columns or race_col not in working.columns:
        return pd.DataFrame()
    working = working[working[time_col].notna() & (working[time_col] > 0)]
    working["pace_per_km"] = working[time_col].apply(lambda t: calculate_pace_per_km(t))
    avg_pace = working.groupby(race_col)["pace_per_km"].mean().reset_index()
    avg_pace.columns = ["Corrida", "Ritmo Medio (min/km)"]
    avg_pace = avg_pace.sort_values("Ritmo Medio (min/km)").reset_index(drop=True)
    return avg_pace


@st.cache_data
def compute_course_characteristics(metadata_df):
    if metadata_df is None or metadata_df.empty:
        return pd.DataFrame()
    working = metadata_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    rows = []
    for race_name in sorted(working[race_col].unique()):
        race_data = working[working[race_col] == race_name]
        if race_data.empty:
            continue
        latest = race_data.iloc[-1]
        cidade = latest.get("city", "")
        pais = latest.get("country", "")
        course_type = latest.get("course_type", "N/A")
        elevation = latest.get("elevation_gain_m", "N/A")
        weather = latest.get("weather_notes", "N/A")
        rows.append({
            "Corrida": race_name,
            "Cidade": cidade,
            "Pais": pais,
            "Tipo de Percurso": course_type,
            "Desnivel (m)": elevation,
            "Clima Tipico": weather,
        })
    descriptions = {
        "Tokyo Marathon": "Percurso plano e rapido pelas ruas de Toquio, conhecido por torcidas entusiastas e excelente organizacao. O trajeto passa por marcos como a Torre de Toquio e o Palacio Imperial.",
        "Boston Marathon": "A maratona anual mais antiga do mundo, com a famosa Heartbreak Hill entre as milhas 20 e 21. Percurso ponto a ponto de Hopkinton ate Boston com criterios rigorosos de qualificacao.",
        "London Marathon": "Percurso predominantemente plano que serpenteia pelo centro de Londres passando por marcos iconicos como a Tower Bridge, o London Eye e o Palacio de Buckingham. Conhecida pelo enorme arrecadamento para caridade.",
        "Berlin Marathon": "O mais plano e rapido dos seis majors, Berlin produziu diversos recordes mundiais. O percurso passa pelo Portao de Brandemburgo e outros sitios historicos.",
        "Chicago Marathon": "Percurso plano e rapido em loop pelos bairros diversos de Chicago com vista para o skyline da cidade e o Lago Michigan. Condicoes climaticas favoraveis no outono frequentemente resultam em tempos rapidos.",
        "New York City Marathon": "A maior maratona do mundo com um percurso desafiador por todos os cinco distritos de Nova York. Atravessa pontes e terrenos variados, tornando-a uma das mais dificeis entre os majors.",
        "Tokyo": "Percurso plano e rapido pelas ruas de Toquio, conhecido por torcidas entusiastas e excelente organizacao. O trajeto passa por marcos como a Torre de Toquio e o Palacio Imperial.",
        "Boston": "A maratona anual mais antiga do mundo, com a famosa Heartbreak Hill entre as milhas 20 e 21. Percurso ponto a ponto de Hopkinton ate Boston com criterios rigorosos de qualificacao.",
        "London": "Percurso predominantemente plano que serpenteia pelo centro de Londres passando por marcos iconicos como a Tower Bridge, o London Eye e o Palacio de Buckingham. Conhecida pelo enorme arrecadamento para caridade.",
        "Berlin": "O mais plano e rapido dos seis majors, Berlin produziu diversos recordes mundiais. O percurso passa pelo Portao de Brandemburgo e outros sitios historicos.",
        "Chicago": "Percurso plano e rapido em loop pelos bairros diversos de Chicago com vista para o skyline da cidade e o Lago Michigan. Condicoes climaticas favoraveis no outono frequentemente resultam em tempos rapidos.",
        "New York City": "A maior maratona do mundo com um percurso desafiador por todos os cinco distritos de Nova York. Atravessa pontes e terrenos variados, tornando-a uma das mais dificeis entre os majors.",
    }
    desc_df = pd.DataFrame(rows)
    if not desc_df.empty:
        desc_df["Descricao do Percurso"] = desc_df["Corrida"].map(descriptions)
    return desc_df


@st.cache_data
def filter_results_data(results_df, years, genders, races):
    if results_df is None or results_df.empty:
        return results_df
    working = results_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    if "year" in working.columns and years:
        working = working[working["year"].isin(years)]
    if "gender" in working.columns and genders:
        working = working[working["gender"].isin(genders)]
    if race_col in working.columns and races:
        working = working[working[race_col].isin(races)]
    return working


@st.cache_data
def filter_winners_data(winners_df, years, genders, races):
    if winners_df is None or winners_df.empty:
        return winners_df
    working = winners_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    if "year" in working.columns and years:
        working = working[working["year"].isin(years)]
    if "gender" in working.columns and genders:
        working = working[working["gender"].isin(genders)]
    if race_col in working.columns and races:
        working = working[working[race_col].isin(races)]
    return working


@st.cache_data
def filter_metadata_data(metadata_df, years, races):
    if metadata_df is None or metadata_df.empty:
        return metadata_df
    working = metadata_df.copy()
    race_col = "marathon" if "marathon" in working.columns else "race"
    if "year" in working.columns and years:
        working = working[working["year"].isin(years)]
    if race_col in working.columns and races:
        working = working[working[race_col].isin(races)]
    return working


def render_race_comparison_page(results_df, winners_df, metadata_df):
    st.title("Comparacao de Corridas")
    st.markdown(
        "Compare as seis maratonas do Abbott World Marathon Majors em multiplas dimensoes, "
        "incluindo tempos de chegada, ritmo, participacao e caracteristicas do percurso. "
        "Utilize os filtros abaixo para concentrar sua analise em anos, generos e corridas especificas."
    )
    st.divider()

    available_years = YEARS
    if results_df is not None and not results_df.empty and "year" in results_df.columns:
        available_years = sorted(results_df["year"].dropna().unique().tolist())
    available_genders = ["M", "F"]
    if results_df is not None and not results_df.empty and "gender" in results_df.columns:
        available_genders = sorted(results_df["gender"].dropna().unique().tolist())
    available_races = RACE_NAMES
    race_col = "marathon" if (results_df is not None and "marathon" in results_df.columns) else "race"
    if results_df is not None and not results_df.empty and race_col in results_df.columns:
        available_races = sorted(results_df[race_col].dropna().unique().tolist())

    with st.expander("Filtros", expanded=True):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            selected_years = st.multiselect(
                "Ano",
                options=available_years,
                default=available_years,
                key="race_comp_year",
            )
        with filter_col2:
            selected_genders = st.multiselect(
                "Genero",
                options=available_genders,
                default=available_genders,
                key="race_comp_gender",
            )
        with filter_col3:
            selected_races = st.multiselect(
                "Corrida",
                options=available_races,
                default=available_races,
                key="race_comp_race",
            )

    filtered_results = filter_results_data(results_df, selected_years, selected_genders, selected_races)
    filtered_winners = filter_winners_data(winners_df, selected_years, selected_genders, selected_races)
    filtered_metadata = filter_metadata_data(metadata_df, selected_years, selected_races)

    st.divider()

    st.markdown("### Visao Geral das Corridas")
    summary_table = compute_race_summary_table(results_df, selected_years, selected_genders, selected_races)
    if not summary_table.empty:
        st.dataframe(summary_table, use_container_width=True, height=min(400, 50 + len(summary_table) * 40))
    else:
        st.info("Nenhum dado disponivel para os filtros selecionados.")

    st.divider()

    st.markdown("### Visualizacoes")
    viz_row1_col1, viz_row1_col2 = st.columns(2)
    viz_row2_col1, viz_row2_col2 = st.columns(2)

    with viz_row1_col1:
        st.markdown("#### Evolucao dos Tempos dos Vencedores")
        if filtered_winners is not None and not filtered_winners.empty:
            fig_evolution = plot_winner_time_evolution(filtered_winners, gender="All")
            fig_evolution.update_layout(
                title=dict(text="Evolucao dos Tempos Vencedores por Corrida"),
            )
            st.plotly_chart(fig_evolution, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Dados de vencedores nao disponiveis"), use_container_width=True)

    with viz_row1_col2:
        st.markdown("#### Distribuicao dos Tempos por Corrida")
        if filtered_results is not None and not filtered_results.empty:
            fig_boxplot = plot_boxplot_times_by_race(filtered_results)
            fig_boxplot.update_layout(
                title=dict(text="Distribuicao dos Tempos de Chegada por Corrida"),
            )
            st.plotly_chart(fig_boxplot, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Dados de resultados nao disponiveis"), use_container_width=True)

    with viz_row2_col1:
        st.markdown("#### Ranking das Corridas Mais Rapidas")
        ranking_df = compute_fastest_marathons_ranking(results_df, selected_years, selected_genders, selected_races)
        if not ranking_df.empty:
            fig_ranking = go.Figure()
            fig_ranking.add_trace(
                go.Bar(
                    x=ranking_df["Tempo Medio (horas)"],
                    y=ranking_df["Corrida"],
                    orientation="h",
                    marker_color=[RACE_COLORS.get(race, COLOR_PALETTE["primary"]) for race in ranking_df["Corrida"]],
                    text=ranking_df["Tempo Medio"],
                    textposition="outside",
                )
            )
            fig_ranking.update_layout(
                xaxis_title="Tempo Medio de Chegada (horas)",
                yaxis_title="Corrida",
                yaxis=dict(autorange="reversed"),
            )
            fig_ranking = apply_standard_layout(fig_ranking, title="Ranking: Corridas Mais Rapidas para Mais Lentas")
            st.plotly_chart(fig_ranking, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Sem dados disponiveis para o ranking"), use_container_width=True)

    with viz_row2_col2:
        st.markdown("#### Comparacao por Genero em Cada Corrida")
        if filtered_results is not None and not filtered_results.empty:
            fig_gender = plot_gender_comparison(filtered_results)
            fig_gender.update_layout(
                title=dict(text="Comparacao de Genero por Corrida"),
                xaxis_title="Corrida",
                yaxis_title="Tempo Medio de Chegada (horas)",
                legend_title="Genero",
            )
            st.plotly_chart(fig_gender, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Dados de resultados nao disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Analise de Ritmo por Corrida")
    pace_stats = compute_pace_statistics(results_df, selected_years, selected_genders, selected_races)
    avg_pace_by_race = compute_avg_pace_by_race(results_df, selected_years, selected_genders, selected_races)

    if not avg_pace_by_race.empty:
        fig_pace = go.Figure()
        fig_pace.add_trace(
            go.Bar(
                x=avg_pace_by_race["Corrida"],
                y=avg_pace_by_race["Ritmo Medio (min/km)"],
                marker_color=[RACE_COLORS.get(race, COLOR_PALETTE["secondary"]) for race in avg_pace_by_race["Corrida"]],
                text=[format_pace_str(p) for p in avg_pace_by_race["Ritmo Medio (min/km)"]],
                textposition="outside",
            )
        )
        fig_pace.update_layout(
            xaxis_title="Corrida",
            yaxis_title="Ritmo Medio (min/km)",
        )
        fig_pace = apply_standard_layout(fig_pace, title="Ritmo Medio por Quilometro em Cada Corrida")
        st.plotly_chart(fig_pace, use_container_width=True)
    else:
        st.info("Nenhum dado de ritmo disponivel para os filtros selecionados.")

    if not pace_stats.empty:
        st.markdown("#### Estatisticas de Ritmo por Corrida")
        st.dataframe(pace_stats, use_container_width=True, height=min(350, 50 + len(pace_stats) * 40))
    else:
        st.info("Nenhuma estatistica de ritmo disponivel.")

    st.divider()

    st.markdown("### Volume de Participantes")
    participant_volume = compute_participant_volume(results_df, selected_years, selected_genders, selected_races)
    if not participant_volume.empty:
        race_col_vol = "marathon" if "marathon" in participant_volume.columns else "race"
        fig_volume = go.Figure()
        races_in_data = sorted(participant_volume[race_col_vol].unique())
        for race in races_in_data:
            race_data = participant_volume[participant_volume[race_col_vol] == race].sort_values("year")
            fig_volume.add_trace(
                go.Bar(
                    name=race,
                    x=race_data["year"],
                    y=race_data["Finalistas"],
                    marker_color=RACE_COLORS.get(race, COLOR_PALETTE["primary"]),
                )
            )
        fig_volume.update_layout(
            xaxis_title="Ano",
            yaxis_title="Total de Finalistas",
            barmode="group",
            legend_title="Corrida",
        )
        fig_volume = apply_standard_layout(fig_volume, title="Total de Finalistas por Corrida e Ano")
        st.plotly_chart(fig_volume, use_container_width=True)
    else:
        st.info("Nenhum dado de participacao disponivel para os filtros selecionados.")

    st.divider()

    st.markdown("### Impacto da COVID-19 na Participacao")
    if metadata_df is not None and not metadata_df.empty:
        fig_covid = plot_covid_impact(metadata_df)
        fig_covid.update_layout(
            title=dict(text="Impacto da COVID-19 na Participacao (2018-2025)"),
            xaxis_title="Ano",
            yaxis_title="Participantes",
        )
        st.plotly_chart(fig_covid, use_container_width=True)
    elif filtered_metadata is not None and not filtered_metadata.empty:
        fig_covid = plot_covid_impact(filtered_metadata)
        fig_covid.update_layout(
            title=dict(text="Impacto da COVID-19 na Participacao (2018-2025)"),
            xaxis_title="Ano",
            yaxis_title="Participantes",
        )
        st.plotly_chart(fig_covid, use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("Metadados nao disponiveis para analise do impacto da COVID-19"), use_container_width=True)

    st.divider()

    with st.expander("Caracteristicas do Percurso", expanded=False):
        st.markdown("### Caracteristicas do Percurso")
        st.markdown(
            "Detalhes sobre o tipo de percurso, desnivel e condicoes climaticas tipicas de cada "
            "maratona do World Marathon Majors."
        )
        course_chars = compute_course_characteristics(metadata_df)
        if not course_chars.empty:
            display_cols = ["Corrida", "Cidade", "Pais", "Tipo de Percurso", "Desnivel (m)", "Clima Tipico"]
            available_cols = [c for c in display_cols if c in course_chars.columns]
            st.dataframe(
                course_chars[available_cols],
                use_container_width=True,
                height=min(350, 50 + len(course_chars) * 40),
            )
            if "Descricao do Percurso" in course_chars.columns:
                st.markdown("#### Descricoes dos Percursos")
                for _, row in course_chars.iterrows():
                    race_name = row.get("Corrida", "")
                    description = row.get("Descricao do Percurso", "")
                    if race_name and description:
                        st.markdown(f"**{race_name}**: {description}")
        else:
            st.info("Nenhum dado sobre caracteristicas do percurso disponivel.")


if __name__ == "__main__" or "streamlit" in os.path.basename(sys.argv[0]).lower():
    data_loaded = False
    results_df = None
    winners_df = None
    metadata_df = None

    if "marathon_results" in st.session_state and st.session_state["marathon_results"] is not None:
        results_df = st.session_state["marathon_results"]
        data_loaded = True
    elif "results" in st.session_state and st.session_state["results"] is not None:
        results_df = st.session_state["results"]
        data_loaded = True
    elif "combined_data" in st.session_state and st.session_state["combined_data"] is not None:
        results_df = st.session_state["combined_data"]
        data_loaded = True

    if "winners" in st.session_state and st.session_state["winners"] is not None:
        winners_df = st.session_state["winners"]
    elif "winners_data" in st.session_state and st.session_state["winners_data"] is not None:
        winners_df = st.session_state["winners_data"]

    if "metadata" in st.session_state and st.session_state["metadata"] is not None:
        metadata_df = st.session_state["metadata"]
    elif "race_metadata" in st.session_state and st.session_state["race_metadata"] is not None:
        metadata_df = st.session_state["race_metadata"]

    if not data_loaded:
        try:
            from src.data.load_data import load_csv, get_data_filepaths
            filepaths = get_data_filepaths()
            raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
            processed_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

            combined_path = os.path.normpath(os.path.join(processed_dir, "combined_marathon_data.csv"))
            results_path = os.path.normpath(os.path.join(raw_dir, "marathon_results.csv"))
            winners_path = os.path.normpath(os.path.join(raw_dir, "winners_data.csv"))
            metadata_path = os.path.normpath(os.path.join(raw_dir, "race_metadata.csv"))

            if os.path.exists(combined_path):
                results_df = pd.read_csv(combined_path)
                data_loaded = True
            elif os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                data_loaded = True

            if os.path.exists(winners_path):
                winners_df = pd.read_csv(winners_path)

            if os.path.exists(metadata_path):
                metadata_df = pd.read_csv(metadata_path)
        except Exception:
            pass

    if data_loaded or winners_df is not None:
        render_race_comparison_page(results_df, winners_df, metadata_df)
    else:
        st.warning(
            "Nenhum dado carregado. Certifique-se de que os conjuntos de dados de maratona estao "
            "disponiveis no diretorio de dados ou carregue-os pelo ponto de entrada principal do "
            "aplicativo para que sejam armazenados no estado da sessao."
        )
