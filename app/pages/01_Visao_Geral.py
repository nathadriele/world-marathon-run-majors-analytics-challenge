import os
import sys
import streamlit as st
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "src")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app_utils import load_css

load_css()

from src.visualization.plots import (
    plot_winner_time_evolution,
    plot_finish_time_distribution,
    create_empty_figure,
    COLOR_PALETTE,
    RACE_COLORS,
)

from src.utils.config import (
    MARATHON_DISTANCE_KM,
    RACE_NAMES,
    YEARS,
)

from src.utils.helpers import (
    seconds_to_time,
    calculate_pace_per_km,
    format_pace_str,
)


@st.cache_data
def load_data():
    base_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw"))
    results_df = None
    winners_df = None
    metadata_df = None

    if "marathon_results" in st.session_state and st.session_state["marathon_results"] is not None:
        results_df = st.session_state["marathon_results"]
    elif "results" in st.session_state and st.session_state["results"] is not None:
        results_df = st.session_state["results"]
    else:
        results_path = os.path.join(base_dir, "marathon_results.csv")
        if os.path.exists(results_path):
            results_df = pd.read_csv(results_path)

    if "winners" in st.session_state and st.session_state["winners"] is not None:
        winners_df = st.session_state["winners"]
    elif "winners_data" in st.session_state and st.session_state["winners_data"] is not None:
        winners_df = st.session_state["winners_data"]
    else:
        winners_path = os.path.join(base_dir, "winners_data.csv")
        if os.path.exists(winners_path):
            winners_df = pd.read_csv(winners_path)

    if "metadata" in st.session_state and st.session_state["metadata"] is not None:
        metadata_df = st.session_state["metadata"]
    elif "race_metadata" in st.session_state and st.session_state["race_metadata"] is not None:
        metadata_df = st.session_state["race_metadata"]
    else:
        metadata_path = os.path.join(base_dir, "race_metadata.csv")
        if os.path.exists(metadata_path):
            metadata_df = pd.read_csv(metadata_path)

    return results_df, winners_df, metadata_df


@st.cache_data
def compute_kpi_metrics(results_df, winners_df, metadata_df):
    metrics = {}

    if results_df is not None and not results_df.empty:
        race_col = "marathon" if "marathon" in results_df.columns else "race"
        total_races = results_df[race_col].nunique() if race_col in results_df.columns else len(RACE_NAMES)
        total_years = results_df["year"].nunique() if "year" in results_df.columns else len(YEARS)

        if "status" in results_df.columns:
            total_finishers = len(results_df[results_df["status"] == "Finished"])
        else:
            total_finishers = len(results_df)

        country_col = "country" if "country" in results_df.columns else "nationality"
        total_countries = results_df[country_col].nunique() if country_col in results_df.columns else 0

        time_col = "finish_time_sec" if "finish_time_sec" in results_df.columns else "finish_seconds"
        valid_times = results_df[results_df[time_col].notna() & (results_df[time_col] > 0)]

        if len(valid_times) > 0:
            male_mask = valid_times["gender"] == "M" if "gender" in valid_times.columns else pd.Series([False] * len(valid_times))
            female_mask = valid_times["gender"] == "F" if "gender" in valid_times.columns else pd.Series([False] * len(valid_times))

            best_male_time_sec = valid_times.loc[male_mask, time_col].min() if male_mask.any() else 0
            best_female_time_sec = valid_times.loc[female_mask, time_col].min() if female_mask.any() else 0

            best_male_row = valid_times.loc[valid_times[time_col] == best_male_time_sec].iloc[0] if male_mask.any() and best_male_time_sec > 0 else None
            best_female_row = valid_times.loc[valid_times[time_col] == best_female_time_sec].iloc[0] if female_mask.any() and best_female_time_sec > 0 else None

            best_male_pace = calculate_pace_per_km(best_male_time_sec) if best_male_time_sec > 0 else 0
            best_female_pace = calculate_pace_per_km(best_female_time_sec) if best_female_time_sec > 0 else 0
            avg_finishers_per_race = total_finishers / total_races if total_races > 0 else 0
        else:
            best_male_time_sec = 0
            best_female_time_sec = 0
            best_male_row = None
            best_female_row = None
            best_male_pace = 0
            best_female_pace = 0
            avg_finishers_per_race = 0

        brazil_col = "country" if "country" in results_df.columns else "nationality"
        brazil_runners = len(results_df[results_df[brazil_col] == "BRA"]) if brazil_col in results_df.columns else 0

        metrics["total_races"] = total_races
        metrics["total_years"] = total_years
        metrics["total_finishers"] = total_finishers
        metrics["total_countries"] = total_countries
        metrics["best_male_time_sec"] = best_male_time_sec if not pd.isna(best_male_time_sec) else 0
        metrics["best_female_time_sec"] = best_female_time_sec if not pd.isna(best_female_time_sec) else 0
        metrics["best_male_row"] = best_male_row
        metrics["best_female_row"] = best_female_row
        metrics["best_male_pace"] = best_male_pace
        metrics["best_female_pace"] = best_female_pace
        metrics["avg_finishers_per_race"] = avg_finishers_per_race
        metrics["brazil_runners"] = brazil_runners

    if winners_df is not None and not winners_df.empty:
        winner_country_col = "winner_country" if "winner_country" in winners_df.columns else "country"
        if winner_country_col in winners_df.columns:
            total_wins = len(winners_df)
            kenya_wins = len(winners_df[winners_df[winner_country_col] == "KEN"])
            metrics["kenya_dominance_pct"] = (kenya_wins / total_wins * 100) if total_wins > 0 else 0
            country_counts = winners_df[winner_country_col].value_counts()
            metrics["most_dominant_country"] = country_counts.index[0] if len(country_counts) > 0 else "N/A"
            metrics["kenya_wins"] = kenya_wins
            metrics["total_winner_records"] = total_wins

        win_time_col = "winning_time_sec" if "winning_time_sec" in winners_df.columns else None
        win_race_col = "marathon" if "marathon" in winners_df.columns else "race"

        if win_time_col and win_race_col in winners_df.columns and "year" in winners_df.columns:
            avg_by_race = winners_df.groupby(win_race_col)[win_time_col].mean()
            if len(avg_by_race) > 0:
                metrics["fastest_course"] = avg_by_race.idxmin()
                metrics["fastest_avg_time_sec"] = avg_by_race.min()

        covid_years = [2020, 2021]
        if "year" in winners_df.columns:
            pre_covid = winners_df[~winners_df["year"].isin(covid_years)]
            covid_data = winners_df[winners_df["year"].isin(covid_years)]
            if win_time_col and len(pre_covid) > 0 and len(covid_data) > 0:
                metrics["pre_covid_avg_time"] = pre_covid[win_time_col].mean()
                metrics["covid_avg_time"] = covid_data[win_time_col].mean()
                metrics["covid_races_held"] = len(covid_data)

    if metadata_df is not None and not metadata_df.empty:
        if "participants_estimate" in metadata_df.columns:
            metrics["total_participants_estimate"] = metadata_df["participants_estimate"].sum()
        if "status" in metadata_df.columns:
            cancelled = metadata_df[metadata_df["status"] == "Cancelled"]
            metrics["cancelled_races"] = len(cancelled)

    return metrics


@st.cache_data
def compute_fastest_course(winners_df):
    if winners_df is None or winners_df.empty:
        return None, None
    time_col = "winning_time_sec" if "winning_time_sec" in winners_df.columns else None
    race_col = "marathon" if "marathon" in winners_df.columns else "race"
    if time_col is None or race_col not in winners_df.columns:
        return None, None
    avg_by_race = winners_df.groupby(race_col)[time_col].mean()
    if len(avg_by_race) == 0:
        return None, None
    fastest_race = avg_by_race.idxmin()
    fastest_avg = avg_by_race.min()
    return fastest_race, fastest_avg


@st.cache_data
def compute_covid_impact(winners_df):
    if winners_df is None or winners_df.empty:
        return None, None, 0
    time_col = "winning_time_sec" if "winning_time_sec" in winners_df.columns else None
    if time_col is None or "year" not in winners_df.columns:
        return None, None, 0
    covid_years = [2020, 2021]
    pre_covid = winners_df[~winners_df["year"].isin(covid_years)]
    covid_data = winners_df[winners_df["year"].isin(covid_years)]
    pre_avg = pre_covid[time_col].mean() if len(pre_covid) > 0 else None
    covid_avg = covid_data[time_col].mean() if len(covid_data) > 0 else None
    races_held = len(covid_data)
    return pre_avg, covid_avg, races_held


@st.cache_data
def compute_kenya_stats(winners_df):
    if winners_df is None or winners_df.empty:
        return 0, 0, 0.0
    country_col = "winner_country" if "winner_country" in winners_df.columns else "country"
    if country_col not in winners_df.columns:
        return 0, 0, 0.0
    total = len(winners_df)
    kenya = len(winners_df[winners_df[country_col] == "KEN"])
    pct = (kenya / total * 100) if total > 0 else 0.0
    return total, kenya, pct


def render_overview_page(results_df, winners_df, metadata_df):
    st.title("Visao Geral")
    st.markdown(
        "Painel interativo para analise completa das Maratonas Majores do Mundo (Abbott World Marathon Majors) "
        "-- Tokyo, Boston, Londres, Berlim, Chicago e Nova York. Explore resultados de 2018 a 2025 com "
        "dados detalhados de desempenho, vencedores e estatisticas globais."
    )

    with st.expander("Filtros"):
        filter_col1, filter_col2, filter_col3 = st.columns(3)
        with filter_col1:
            anos_disponiveis = sorted(YEARS)
            selected_ano = st.multiselect("Ano", anos_disponiveis, default=anos_disponiveis)
        with filter_col2:
            corridas_disponiveis = list(RACE_NAMES)
            selected_corrida = st.multiselect("Corrida", corridas_disponiveis, default=corridas_disponiveis)
        with filter_col3:
            genero_options = ["Todos", "M", "F"]
            selected_genero = st.selectbox("Genero", genero_options, index=0)

    filtered_results = results_df
    if filtered_results is not None and not filtered_results.empty:
        if selected_ano and "year" in filtered_results.columns:
            filtered_results = filtered_results[filtered_results["year"].isin(selected_ano)]
        if selected_corrida:
            race_col = "marathon" if "marathon" in filtered_results.columns else "race"
            if race_col in filtered_results.columns:
                filtered_results = filtered_results[filtered_results[race_col].isin(selected_corrida)]
        if selected_genero != "Todos" and "gender" in filtered_results.columns:
            filtered_results = filtered_results[filtered_results["gender"] == selected_genero]

    filtered_winners = winners_df
    if filtered_winners is not None and not filtered_winners.empty:
        if selected_ano and "year" in filtered_winners.columns:
            filtered_winners = filtered_winners[filtered_winners["year"].isin(selected_ano)]
        if selected_corrida:
            race_col = "marathon" if "marathon" in filtered_winners.columns else "race"
            if race_col in filtered_winners.columns:
                filtered_winners = filtered_winners[filtered_winners[race_col].isin(selected_corrida)]
        if selected_genero != "Todos" and "gender" in filtered_winners.columns:
            filtered_winners = filtered_winners[filtered_winners["gender"] == selected_genero]

    filtered_metadata = metadata_df
    if filtered_metadata is not None and not filtered_metadata.empty:
        if selected_ano and "year" in filtered_metadata.columns:
            filtered_metadata = filtered_metadata[filtered_metadata["year"].isin(selected_ano)]
        if selected_corrida:
            m_race_col = "marathon" if "marathon" in filtered_metadata.columns else "race"
            if m_race_col in filtered_metadata.columns:
                filtered_metadata = filtered_metadata[filtered_metadata[m_race_col].isin(selected_corrida)]

    metrics = compute_kpi_metrics(filtered_results, filtered_winners, filtered_metadata)

    st.markdown("### Indicadores Principais de Desempenho")

    row1_col1, row1_col2, row1_col3, row1_col4 = st.columns(4)
    with row1_col1:
        st.metric(label="Total de Corridas", value=f"{metrics.get('total_races', 6)}")
    with row1_col2:
        st.metric(label="Anos Analisados", value=f"{metrics.get('total_years', len(YEARS))}", delta="2018 - 2025")
    with row1_col3:
        total_fin = metrics.get("total_finishers", 0)
        if total_fin >= 1_000_000:
            fin_display = f"{total_fin / 1_000_000:.1f}M"
        elif total_fin >= 1_000:
            fin_display = f"{total_fin / 1_000:.1f}K"
        else:
            fin_display = str(total_fin)
        st.metric(label="Total de Finalizadores", value=fin_display)
    with row1_col4:
        st.metric(label="Total de Paises", value=f"{metrics.get('total_countries', 0)}")

    st.markdown("")
    row2_col1, row2_col2, row2_col3, row2_col4 = st.columns(4)
    with row2_col1:
        best_male_sec = metrics.get("best_male_time_sec", 0)
        best_male_time_str = seconds_to_time(best_male_sec)
        best_male_row = metrics.get("best_male_row")
        male_detail = ""
        if best_male_row is not None:
            race_col = "marathon" if "marathon" in best_male_row.index else "race"
            if race_col in best_male_row.index and "year" in best_male_row.index:
                male_detail = f"{best_male_row[race_col]} {int(best_male_row['year'])}"
        st.metric(
            label="Melhor Tempo Masculino",
            value=best_male_time_str,
            delta=male_detail if male_detail else None,
        )
    with row2_col2:
        best_female_sec = metrics.get("best_female_time_sec", 0)
        best_female_time_str = seconds_to_time(best_female_sec)
        best_female_row = metrics.get("best_female_row")
        female_detail = ""
        if best_female_row is not None:
            race_col = "marathon" if "marathon" in best_female_row.index else "race"
            if race_col in best_female_row.index and "year" in best_female_row.index:
                female_detail = f"{best_female_row[race_col]} {int(best_female_row['year'])}"
        st.metric(
            label="Melhor Tempo Feminino",
            value=best_female_time_str,
            delta=female_detail if female_detail else None,
        )
    with row2_col3:
        best_male_pace = metrics.get("best_male_pace", 0)
        male_pace_str = format_pace_str(best_male_pace) + " /km"
        st.metric(label="Melhor Ritmo Masculino", value=male_pace_str)
    with row2_col4:
        best_female_pace = metrics.get("best_female_pace", 0)
        female_pace_str = format_pace_str(best_female_pace) + " /km"
        st.metric(label="Melhor Ritmo Feminino", value=female_pace_str)

    st.markdown("")
    row3_col1, row3_col2, row3_col3, row3_col4 = st.columns(4)
    with row3_col1:
        st.metric(
            label="Distancia da Maratona",
            value=f"{MARATHON_DISTANCE_KM} km",
            delta="26.2188 milhas",
        )
    with row3_col2:
        avg_fin = metrics.get("avg_finishers_per_race", 0)
        if avg_fin >= 1_000:
            avg_display = f"{avg_fin / 1_000:.1f}K"
        else:
            avg_display = f"{avg_fin:,.0f}"
        st.metric(label="Media de Finalizadores por Corrida", value=avg_display)
    with row3_col3:
        dominant = metrics.get("most_dominant_country", "KEN")
        kenya_wins = metrics.get("kenya_wins", 0)
        total_w = metrics.get("total_winner_records", 0)
        dominance_str = "Quenia" if dominant in ("KEN", "Kenya") else str(dominant)
        st.metric(
            label="Pais Mais Dominante",
            value=dominance_str,
            delta=f"{kenya_wins} vitorias de {total_w}" if total_w > 0 else None,
        )
    with row3_col4:
        brazil_count = metrics.get("brazil_runners", 0)
        if brazil_count >= 1_000:
            brazil_display = f"{brazil_count / 1_000:.1f}K"
        else:
            brazil_display = str(brazil_count)
        st.metric(label="Total de Corredores Brasileiros", value=brazil_display)

    st.divider()

    st.markdown("### Insights Rapidos")
    insight_col1, insight_col2, insight_col3 = st.columns(3)

    with insight_col1:
        fastest_race, fastest_avg = compute_fastest_course(filtered_winners)
        with st.expander("Percurso Mais Rapido", expanded=True):
            if fastest_race and fastest_avg:
                fastest_time_str = seconds_to_time(fastest_avg)
                st.markdown(
                    f"O **{fastest_race}** possui o menor tempo medio de vitoria entre todas as "
                    f"Maratonas Majores do Mundo de 2018 a 2025, com tempo medio de vitoria de "
                    f"**{fastest_time_str}**. O percurso plano e rapido de Berlim historicamente "
                    f"produziu recordes mundiais, tornando-o o local principal para "
                    f"corredores de elite em busca de melhores marcas pessoais e mundiais."
                )
            else:
                st.info("Dados de vencedores nao disponiveis para calcular o percurso mais rapido.")

    with insight_col2:
        pre_covid_avg, covid_avg, covid_races = compute_covid_impact(filtered_winners)
        with st.expander("Impacto da COVID-19", expanded=True):
            if pre_covid_avg and covid_avg:
                pre_str = seconds_to_time(pre_covid_avg)
                cov_str = seconds_to_time(covid_avg)
                diff_sec = covid_avg - pre_covid_avg
                diff_str = f"+{seconds_to_time(abs(diff_sec))}" if diff_sec > 0 else f"-{seconds_to_time(abs(diff_sec))}"
                cancelled_count = metrics.get("cancelled_races", 0)
                st.markdown(
                    f"A pandemia de COVID-19 afetou severamente o calendario de maratonas em 2020-2021. "
                    f"**{cancelled_count} maratonas maiores foram canceladas** durante esse periodo. "
                    f"As corridas que ocorreram tiveram campos reduzidos e escalas diferentes. "
                    f"O tempo medio de vitoria durante os anos de COVID foi **{cov_str}** comparado a "
                    f"**{pre_str}** nos anos sem COVID, uma diferenca de **{diff_str}**. "
                    f"Apenas **{covid_races}** corridas de elite foram realizadas em 2020-2021."
                )
            else:
                st.info("Dados de vencedores nao disponiveis para avaliar o impacto da COVID-19.")

    with insight_col3:
        total_wins, kenya_wins, kenya_pct = compute_kenya_stats(filtered_winners)
        with st.expander("Dominio do Quenia", expanded=True):
            if total_wins > 0:
                st.markdown(
                    f"Atletas quenianos dominaram as Maratonas Majores do Mundo, vencendo "
                    f"**{kenya_wins} de {total_wins}** corridas de elite ({kenya_pct:.1f}% de vitorias) "
                    f"de 2018 a 2025. Essa extraordinaria consistencia em todos os seis percursos "
                    f"e ambos os generos reforca a posicao do Quenia como a nacao preeminente no "
                    f"atletismo de longa distancia. Atletas etiopes fornecem a principal concorrencia, "
                    f"criando uma rivalidade leste-africana convincente no nivel mais alto do esporte."
                )
                st.progress(
                    min(kenya_pct / 100, 1.0),
                    text=f"Taxa de Vitorias do Quenia: {kenya_pct:.1f}%",
                )
            else:
                st.info("Dados de vencedores nao disponiveis para avaliar o dominio do Quenia.")

    st.divider()

    st.markdown("### Visualizacoes Principais")
    viz_col1, viz_col2 = st.columns(2)

    with viz_col1:
        st.markdown("#### Evolucao dos Tempos dos Vencedores")
        if filtered_winners is not None and not filtered_winners.empty:
            fig_evolution = plot_winner_time_evolution(filtered_winners, gender="All")
            st.plotly_chart(fig_evolution, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Dados de vencedores nao disponiveis"), use_container_width=True)

    with viz_col2:
        st.markdown("#### Distribuicao dos Tempos de Chegada")
        if filtered_results is not None and not filtered_results.empty:
            fig_distribution = plot_finish_time_distribution(filtered_results, gender="All")
            st.plotly_chart(fig_distribution, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Dados de resultados nao disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Sobre a Distancia da Maratona")
    st.markdown(
        "A distancia oficial da maratona de **42,195 quilometros (26,2188 milhas)** foi padronizada "
        "pela Associacao Internacional de Federacoes de Atletismo (IAAF) em 1921. Esta distancia "
        "originou-se nas Olimpiadas de Londres de 1908, quando o percurso da maratona foi estendido de "
        "aproximadamente 40 km para 42,195 km para permitir que a corrida terminasse em frente ao "
        "Camara Real no Estadio Olimpico. Hoje, todo percurso de maratona certificado e medido com "
        "precisao de 42,195 km, com tolerancia de nao mais que 0,1% (42 metros) para considerar a "
        "linha de corrida mais curta possivel."
    )


if __name__ == "__main__" or "streamlit" in os.path.basename(sys.argv[0]).lower():
    results_df, winners_df, metadata_df = load_data()

    if results_df is not None or winners_df is not None:
        render_overview_page(results_df, winners_df, metadata_df)
    else:
        st.warning(
            "Nenhum dado carregado. Certifique-se de que os conjuntos de dados de maratona estao "
            "disponiveis no diretorio de dados ou carregue-os atraves do ponto de entrada principal "
            "do aplicativo para que sejam armazenados no estado da sessao."
        )
