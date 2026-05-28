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
    plot_brazil_vs_world,
    plot_brazil_performance,
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
def compute_brazil_data(results_df):
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    country_col = "country" if "country" in results_df.columns else "nationality"
    if country_col not in results_df.columns:
        return pd.DataFrame()
    time_col = "finish_time_sec" if "finish_time_sec" in results_df.columns else "finish_seconds"
    status_col = "status" if "status" in results_df.columns else None
    finished = results_df.copy()
    if status_col and status_col in finished.columns:
        finished = finished[finished[status_col] == "Finished"]
    brazil_df = finished[finished[country_col] == "BRA"].copy()
    if time_col in brazil_df.columns:
        brazil_df = brazil_df[brazil_df[time_col].notna() & (brazil_df[time_col] > 0)]
        brazil_df["pace_per_km"] = brazil_df[time_col].apply(
            lambda x: calculate_pace_per_km(x)
        )
    return brazil_df


def render_brazil_page(results_df, brazil_df):
    st.title("Análise do Brasil")
    st.markdown(
        "Esta pagina apresenta uma analise detalhada da participacao e desempenho "
        "dos corredores brasileiros nas Maratonas Majors do Mundo Abbott. Explore como "
        "os atletas do Brasil competiram em Tokyo, Boston, Londres, Berlim, Chicago e "
        "Nova York entre 2018 e 2025, incluindo tendencias de participacao, comparacoes "
        "de desempenho e conquistas notaveis."
    )

    if brazil_df is None or brazil_df.empty:
        if results_df is not None and not results_df.empty:
            brazil_df = compute_brazil_data(results_df)
        if brazil_df is None or brazil_df.empty:
            st.warning(
                "Nenhum dado de corredor brasileiro disponivel. Verifique se os datasets "
                "de maratona estao carregados e contem registros de finalistas brasileiros."
            )
            return

    time_col = "finish_time_sec" if "finish_time_sec" in brazil_df.columns else "finish_seconds"
    race_col = "marathon" if "marathon" in brazil_df.columns else "race"
    gender_col = "gender" if "gender" in brazil_df.columns else "sex"
    pace_col = "pace_per_km" if "pace_per_km" in brazil_df.columns else None

    total_brazilian_finishers = len(brazil_df)

    avg_finish_time_sec = 0
    if time_col in brazil_df.columns:
        avg_finish_time_sec = brazil_df[time_col].mean()

    best_time_sec = 0
    if time_col in brazil_df.columns:
        best_time_sec = brazil_df[time_col].min()

    most_popular_race = ""
    if race_col in brazil_df.columns:
        race_counts = brazil_df[race_col].value_counts()
        if len(race_counts) > 0:
            most_popular_race = race_counts.index[0]

    avg_pace = 0.0
    if pace_col and pace_col in brazil_df.columns:
        avg_pace = brazil_df[pace_col].mean()
    elif time_col in brazil_df.columns and avg_finish_time_sec > 0:
        avg_pace = calculate_pace_per_km(avg_finish_time_sec)

    total_finishers_all = 0
    if results_df is not None and not results_df.empty:
        status_col = "status" if "status" in results_df.columns else None
        all_finished = results_df.copy()
        if status_col and status_col in all_finished.columns:
            all_finished = all_finished[all_finished[status_col] == "Finished"]
        total_finishers_all = len(all_finished)

    brazilian_share = 0.0
    if total_finishers_all > 0 and total_brazilian_finishers > 0:
        brazilian_share = (total_brazilian_finishers / total_finishers_all) * 100

    st.markdown("### Indicadores Principais de Desempenho")
    row1_col1, row1_col2, row1_col3 = st.columns(3)

    with row1_col1:
        if total_brazilian_finishers >= 1000:
            finishers_display = f"{total_brazilian_finishers / 1000:.1f}K"
        else:
            finishers_display = str(total_brazilian_finishers)
        st.metric(label="Total de Finalizadores Brasileiros", value=finishers_display)

    with row1_col2:
        avg_time_str = seconds_to_time(avg_finish_time_sec)
        st.metric(label="Tempo Medio de Chegada", value=avg_time_str)

    with row1_col3:
        best_time_str = seconds_to_time(best_time_sec)
        st.metric(label="Melhor Tempo Brasileiro", value=best_time_str)

    st.markdown("")
    row2_col1, row2_col2, row2_col3 = st.columns(3)

    with row2_col1:
        st.metric(label="Corrida Mais Popular", value=most_popular_race)

    with row2_col2:
        avg_pace_str = format_pace_str(avg_pace) + " /km"
        st.metric(label="Ritmo Medio", value=avg_pace_str)

    with row2_col3:
        st.metric(
            label="Participacao Brasileira (% do Total)",
            value=f"{brazilian_share:.2f}%"
        )

    st.divider()

    st.markdown("### Participacao Brasileira ao Longo do Tempo")
    if "year" in brazil_df.columns and time_col in brazil_df.columns:
        yearly_counts = brazil_df.groupby("year").size().reset_index(name="finishers")
        yearly_counts = yearly_counts.sort_values("year")

        fig_participation = go.Figure()
        fig_participation.add_trace(
            go.Scatter(
                x=yearly_counts["year"],
                y=yearly_counts["finishers"],
                mode="lines+markers",
                name="Finalizadores Brasileiros",
                line=dict(color=COLOR_PALETTE["success"], width=3),
                marker=dict(size=8, color=COLOR_PALETTE["success"]),
                fill="tozeroy",
                fillcolor="rgba(34, 197, 94, 0.1)",
            )
        )

        covid_years = [2020, 2021]
        for covid_year in covid_years:
            if covid_year in yearly_counts["year"].values:
                covid_count = yearly_counts[yearly_counts["year"] == covid_year]["finishers"].values[0]
                fig_participation.add_annotation(
                    x=covid_year,
                    y=covid_count,
                    text="COVID-19",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(size=10, color=COLOR_PALETTE["warning"]),
                    arrowcolor=COLOR_PALETTE["warning"],
                    ay=-30,
                )
            else:
                fig_participation.add_annotation(
                    x=covid_year,
                    y=0,
                    text="COVID-19",
                    showarrow=True,
                    arrowhead=2,
                    font=dict(size=10, color=COLOR_PALETTE["warning"]),
                    arrowcolor=COLOR_PALETTE["warning"],
                    ay=30,
                )

        for yr in covid_years:
            fig_participation.add_vrect(
                x0=yr - 0.4,
                x1=yr + 0.4,
                fillcolor=COLOR_PALETTE["warning"],
                opacity=0.12,
                line_width=0,
            )

        fig_participation.update_layout(
            xaxis_title="Ano",
            yaxis_title="Numero de Finalizadores Brasileiros",
        )
        fig_participation.update_layout(
            title=dict(
                text="Finalizadores Brasileiros ao Longo dos Anos",
                font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                x=0.5,
                xanchor="center",
            ),
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
            hoverlabel=dict(
                bgcolor=COLOR_PALETTE["card_bg"],
                font_size=12,
                font_color=COLOR_PALETTE["text"],
            ),
        )
        st.plotly_chart(fig_participation, use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("Dados de ano ou tempo de chegada nao disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Presenca Brasileira por Maratona")
    col_race_bar, col_race_pie = st.columns(2)

    with col_race_bar:
        if race_col in brazil_df.columns:
            race_counts = brazil_df[race_col].value_counts().reset_index()
            race_counts.columns = ["race", "finishers"]
            race_counts = race_counts.sort_values("finishers", ascending=False)

            fig_race_bar = go.Figure()
            fig_race_bar.add_trace(
                go.Bar(
                    x=race_counts["race"],
                    y=race_counts["finishers"],
                    marker_color=[RACE_COLORS.get(race, COLOR_PALETTE["primary"]) for race in race_counts["race"]],
                    text=race_counts["finishers"],
                    textposition="outside",
                )
            )
            fig_race_bar.update_layout(
                xaxis_title="Maratona",
                yaxis_title="Numero de Finalizadores Brasileiros",
            )
            fig_race_bar.update_layout(
                title=dict(
                    text="Finalizadores Brasileiros por Maratona",
                    font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                    x=0.5,
                    xanchor="center",
                ),
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
                hoverlabel=dict(
                    bgcolor=COLOR_PALETTE["card_bg"],
                    font_size=12,
                    font_color=COLOR_PALETTE["text"],
                ),
            )
            st.plotly_chart(fig_race_bar, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Dados de corrida nao disponiveis"), use_container_width=True)

    with col_race_pie:
        if race_col in brazil_df.columns:
            race_counts = brazil_df[race_col].value_counts().reset_index()
            race_counts.columns = ["race", "finishers"]

            fig_race_pie = go.Figure()
            fig_race_pie.add_trace(
                go.Pie(
                    labels=race_counts["race"],
                    values=race_counts["finishers"],
                    marker_colors=[RACE_COLORS.get(race, COLOR_PALETTE["primary"]) for race in race_counts["race"]],
                    hole=0.45,
                    textinfo="label+percent",
                    textposition="outside",
                )
            )
            fig_race_pie.update_layout(
                title=dict(
                    text="Distribuicao Brasileira entre Maratonas",
                    font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                    x=0.5,
                    xanchor="center",
                ),
                template="plotly_white",
                paper_bgcolor=COLOR_PALETTE["card_bg"],
                plot_bgcolor=COLOR_PALETTE["background"],
                font=dict(family="Arial", size=12, color=COLOR_PALETTE["text"]),
                margin=dict(l=60, r=30, t=60, b=50),
                hoverlabel=dict(
                    bgcolor=COLOR_PALETTE["card_bg"],
                    font_size=12,
                    font_color=COLOR_PALETTE["text"],
                ),
            )
            st.plotly_chart(fig_race_pie, use_container_width=True)
        else:
            st.plotly_chart(create_empty_figure("Dados de corrida nao disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Brasil vs Resto do Mundo")

    if results_df is not None and not results_df.empty and time_col in results_df.columns:
        country_col_full = "country" if "country" in results_df.columns else "nationality"
        if country_col_full in results_df.columns:
            full_data = results_df[results_df[time_col].notna() & (results_df[time_col] > 0)].copy()

            brazilian_times = full_data[full_data[country_col_full] == "BRA"][time_col]
            non_brazilian_times = full_data[full_data[country_col_full] != "BRA"][time_col]

            fig_box = go.Figure()

            if len(brazilian_times) > 0:
                fig_box.add_trace(
                    go.Box(
                        y=brazilian_times / 3600,
                        name="Brasil",
                        marker_color=COLOR_PALETTE["success"],
                    )
                )

            if len(non_brazilian_times) > 0:
                sample_non_bra = non_brazilian_times
                if len(non_brazilian_times) > 5000:
                    sample_non_bra = non_brazilian_times.sample(n=5000, random_state=42)
                fig_box.add_trace(
                    go.Box(
                        y=sample_non_bra / 3600,
                        name="Resto do Mundo",
                        marker_color=COLOR_PALETTE["primary"],
                    )
                )

            fig_box.update_layout(
                yaxis_title="Tempo de Chegada (horas)",
            )
            fig_box.update_layout(
                title=dict(
                    text="Tempo de Chegada: Brasil vs Resto do Mundo",
                    font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                    x=0.5,
                    xanchor="center",
                ),
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
                hoverlabel=dict(
                    bgcolor=COLOR_PALETTE["card_bg"],
                    font_size=12,
                    font_color=COLOR_PALETTE["text"],
                ),
            )
            st.plotly_chart(fig_box, use_container_width=True)

            stats_data = []
            if len(brazilian_times) > 0:
                stats_data.append({
                    "Grupo": "Brasil",
                    "Tempo Medio": seconds_to_time(brazilian_times.mean()),
                    "Tempo Mediano": seconds_to_time(brazilian_times.median()),
                    "Desvio Padrao (min)": f"{brazilian_times.std() / 60:.1f}",
                    "Quantidade": len(brazilian_times),
                })
            if len(non_brazilian_times) > 0:
                stats_data.append({
                    "Grupo": "Resto do Mundo",
                    "Tempo Medio": seconds_to_time(non_brazilian_times.mean()),
                    "Tempo Mediano": seconds_to_time(non_brazilian_times.median()),
                    "Desvio Padrao (min)": f"{non_brazilian_times.std() / 60:.1f}",
                    "Quantidade": len(non_brazilian_times),
                })

            if stats_data:
                stats_df = pd.DataFrame(stats_data)
                st.markdown("#### Comparacao Estatistica de Desempenho")
                st.dataframe(stats_df, use_container_width=True, hide_index=True)
        else:
            st.plotly_chart(create_empty_figure("Dados de pais nao disponiveis para comparacao"), use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("Dados de resultados nao disponiveis para comparacao"), use_container_width=True)

    st.divider()

    st.markdown("### Melhores Desempenhos Brasileiros")
    if time_col in brazil_df.columns and "runner_name" in brazil_df.columns:
        top_brazil = brazil_df.nsmallest(20, time_col).copy()

        top_brazil["Posicao"] = range(1, len(top_brazil) + 1)
        top_brazil["Tempo"] = top_brazil[time_col].apply(lambda x: seconds_to_time(x))
        top_brazil["Ritmo"] = top_brazil[time_col].apply(lambda x: format_pace_str(calculate_pace_per_km(x)) + " /km")

        name_col = "runner_name" if "runner_name" in top_brazil.columns else "name"
        race_display_col = race_col if race_col in top_brazil.columns else "race"
        gender_display_col = gender_col if gender_col in top_brazil.columns else "gender"

        display_cols = ["Posicao", name_col, race_display_col, "year", "Tempo", "Ritmo", gender_display_col]
        available_cols = [c for c in display_cols if c in top_brazil.columns]

        if available_cols:
            display_df = top_brazil[available_cols].copy()
            rename_map = {}
            if name_col in display_df.columns:
                rename_map[name_col] = "Nome"
            if race_display_col in display_df.columns:
                rename_map[race_display_col] = "Maratona"
            if gender_display_col in display_df.columns:
                rename_map[gender_display_col] = "Genero"
            if "year" in display_df.columns:
                rename_map["year"] = "Ano"
            display_df = display_df.rename(columns=rename_map)

            if len(display_df) > 0:
                best_row = display_df.iloc[0]
                st.success(
                    f"Melhor desempenho brasileiro: {best_row.get('Nome', 'N/A')} "
                    f"na {best_row.get('Maratona', 'N/A')} {best_row.get('Ano', 'N/A')} "
                    f"com o tempo de {best_row.get('Tempo', 'N/A')}"
                )

            st.dataframe(display_df, use_container_width=True, hide_index=True)
    else:
        st.info("Dados insuficientes para exibir os melhores desempenhos brasileiros.")

    st.divider()

    st.markdown("### Distribuicao por Genero")
    col_gender_pie, col_gender_bar = st.columns(2)

    if gender_col in brazil_df.columns:
        gender_counts = brazil_df[gender_col].value_counts().reset_index()
        gender_counts.columns = ["gender", "count"]
        gender_labels = {"M": "Masculino", "F": "Feminino"}
        gender_colors = {"M": COLOR_PALETTE["primary"], "F": COLOR_PALETTE["accent"]}

        with col_gender_pie:
            fig_gender_pie = go.Figure()
            fig_gender_pie.add_trace(
                go.Pie(
                    labels=[gender_labels.get(g, g) for g in gender_counts["gender"]],
                    values=gender_counts["count"],
                    marker_colors=[gender_colors.get(g, COLOR_PALETTE["muted"]) for g in gender_counts["gender"]],
                    hole=0.45,
                    textinfo="label+percent",
                    textposition="outside",
                )
            )
            fig_gender_pie.update_layout(
                title=dict(
                    text="Corredores Brasileiros por Genero",
                    font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                    x=0.5,
                    xanchor="center",
                ),
                template="plotly_white",
                paper_bgcolor=COLOR_PALETTE["card_bg"],
                plot_bgcolor=COLOR_PALETTE["background"],
                font=dict(family="Arial", size=12, color=COLOR_PALETTE["text"]),
                margin=dict(l=60, r=30, t=60, b=50),
                hoverlabel=dict(
                    bgcolor=COLOR_PALETTE["card_bg"],
                    font_size=12,
                    font_color=COLOR_PALETTE["text"],
                ),
            )
            st.plotly_chart(fig_gender_pie, use_container_width=True)

        with col_gender_bar:
            if race_col in brazil_df.columns:
                gender_by_race = brazil_df.groupby([race_col, gender_col]).size().reset_index(name="count")
                races_sorted = sorted(gender_by_race[race_col].unique())

                fig_gender_race = go.Figure()
                for g_label, g_color in [("M", COLOR_PALETTE["primary"]), ("F", COLOR_PALETTE["accent"])]:
                    g_data = gender_by_race[gender_by_race[gender_col] == g_label]
                    values = []
                    for race in races_sorted:
                        match = g_data[g_data[race_col] == race]
                        values.append(match["count"].values[0] if len(match) > 0 else 0)
                    fig_gender_race.add_trace(
                        go.Bar(
                            name=gender_labels.get(g_label, g_label),
                            x=races_sorted,
                            y=values,
                            marker_color=g_color,
                        )
                    )

                fig_gender_race.update_layout(
                    barmode="group",
                    xaxis_title="Maratona",
                    yaxis_title="Numero de Finalizadores Brasileiros",
                    legend_title="Genero",
                )
                fig_gender_race.update_layout(
                    title=dict(
                        text="Distribuicao por Genero e Maratona",
                        font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                        x=0.5,
                        xanchor="center",
                    ),
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
                    hoverlabel=dict(
                        bgcolor=COLOR_PALETTE["card_bg"],
                        font_size=12,
                        font_color=COLOR_PALETTE["text"],
                    ),
                    legend=dict(
                        bgcolor="rgba(255,255,255,0.8)",
                        bordercolor=COLOR_PALETTE["border"],
                        borderwidth=1,
                        font=dict(color=COLOR_PALETTE["text"]),
                    ),
                )
                st.plotly_chart(fig_gender_race, use_container_width=True)
            else:
                st.plotly_chart(create_empty_figure("Dados de corrida nao disponiveis"), use_container_width=True)
    else:
        with col_gender_pie:
            st.plotly_chart(create_empty_figure("Dados de genero nao disponiveis"), use_container_width=True)
        with col_gender_bar:
            st.plotly_chart(create_empty_figure("Dados de genero nao disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Evolucao do Desempenho Brasileiro")
    if "year" in brazil_df.columns and time_col in brazil_df.columns:
        yearly_avg = brazil_df.groupby("year")[time_col].mean().reset_index()
        yearly_avg.columns = ["year", "avg_time_sec"]
        yearly_avg = yearly_avg.sort_values("year")

        fig_evolution = go.Figure()
        fig_evolution.add_trace(
            go.Scatter(
                x=yearly_avg["year"],
                y=yearly_avg["avg_time_sec"] / 3600,
                mode="lines+markers",
                name="Tempo Medio de Chegada",
                line=dict(color=COLOR_PALETTE["secondary"], width=3),
                marker=dict(size=8, color=COLOR_PALETTE["secondary"]),
            )
        )

        if len(yearly_avg) >= 2:
            x_vals = yearly_avg["year"].values
            y_vals = yearly_avg["avg_time_sec"].values / 3600
            z = np.polyfit(x_vals, y_vals, 1)
            p = np.poly1d(z)
            x_line = np.linspace(x_vals.min(), x_vals.max(), 100)
            fig_evolution.add_trace(
                go.Scatter(
                    x=x_line,
                    y=p(x_line),
                    mode="lines",
                    name="Linha de Tendencia",
                    line=dict(color=COLOR_PALETTE["accent"], width=2, dash="dash"),
                )
            )

        fig_evolution.update_layout(
            xaxis_title="Ano",
            yaxis_title="Tempo Medio de Chegada (horas)",
        )
        fig_evolution.update_layout(
            title=dict(
                text="Evolucao do Tempo Medio de Chegada Brasileiro",
                font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                x=0.5,
                xanchor="center",
            ),
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
            hoverlabel=dict(
                bgcolor=COLOR_PALETTE["card_bg"],
                font_size=12,
                font_color=COLOR_PALETTE["text"],
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLOR_PALETTE["border"],
                borderwidth=1,
                font=dict(color=COLOR_PALETTE["text"]),
            ),
        )
        st.plotly_chart(fig_evolution, use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("Dados de ano ou tempo de chegada nao disponiveis"), use_container_width=True)

    st.divider()

    st.markdown("### Analise de Ritmo Brasileiro")
    if time_col in brazil_df.columns:
        brazil_pace_min = brazil_df[time_col] / MARATHON_DISTANCE_KM / 60

        fig_pace = go.Figure()
        fig_pace.add_trace(
            go.Histogram(
                x=brazil_pace_min,
                nbinsx=40,
                marker_color=COLOR_PALETTE["success"],
                opacity=0.7,
                name="Corredores Brasileiros",
            )
        )

        if results_df is not None and not results_df.empty and time_col in results_df.columns:
            country_col_full = "country" if "country" in results_df.columns else "nationality"
            if country_col_full in results_df.columns:
                non_bra = results_df[
                    (results_df[country_col_full] != "BRA")
                    & (results_df[time_col].notna())
                    & (results_df[time_col] > 0)
                ].copy()
                if len(non_bra) > 0:
                    if len(non_bra) > 5000:
                        non_bra = non_bra.sample(n=5000, random_state=42)
                    world_pace_min = non_bra[time_col] / MARATHON_DISTANCE_KM / 60
                    fig_pace.add_trace(
                        go.Histogram(
                            x=world_pace_min,
                            nbinsx=40,
                            marker_color=COLOR_PALETTE["primary"],
                            opacity=0.4,
                            name="Resto do Mundo",
                        )
                    )

        brazil_avg_pace = brazil_pace_min.mean()
        fig_pace.add_vline(
            x=brazil_avg_pace,
            line_dash="dash",
            line_color=COLOR_PALETTE["success"],
            annotation_text=f"Media Brasil: {brazil_avg_pace:.2f} min/km",
            annotation_position="top left",
        )

        if results_df is not None and not results_df.empty and time_col in results_df.columns:
            country_col_full = "country" if "country" in results_df.columns else "nationality"
            if country_col_full in results_df.columns:
                non_bra_all = results_df[
                    (results_df[country_col_full] != "BRA")
                    & (results_df[time_col].notna())
                    & (results_df[time_col] > 0)
                ].copy()
                if len(non_bra_all) > 0:
                    world_avg_pace = (non_bra_all[time_col] / MARATHON_DISTANCE_KM / 60).mean()
                    fig_pace.add_vline(
                        x=world_avg_pace,
                        line_dash="dot",
                        line_color=COLOR_PALETTE["primary"],
                        annotation_text=f"Media Mundial: {world_avg_pace:.2f} min/km",
                        annotation_position="top right",
                    )

        fig_pace.update_layout(
            xaxis_title="Ritmo (min/km)",
            yaxis_title="Numero de Corredores",
            barmode="overlay",
        )
        fig_pace.update_layout(
            title=dict(
                text="Distribuicao de Ritmo Brasileiro vs Media Mundial",
                font=dict(size=18, color=COLOR_PALETTE["text"], family="Arial"),
                x=0.5,
                xanchor="center",
            ),
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
            hoverlabel=dict(
                bgcolor=COLOR_PALETTE["card_bg"],
                font_size=12,
                font_color=COLOR_PALETTE["text"],
            ),
            legend=dict(
                bgcolor="rgba(255,255,255,0.8)",
                bordercolor=COLOR_PALETTE["border"],
                borderwidth=1,
                font=dict(color=COLOR_PALETTE["text"]),
            ),
        )
        st.plotly_chart(fig_pace, use_container_width=True)
    else:
        st.plotly_chart(create_empty_figure("Dados de tempo de chegada nao disponiveis"), use_container_width=True)

    st.divider()

    with st.expander("Momentos Notaveis do Brasil"):
        st.markdown(
            "#### Marilson Gomes dos Santos -- Campeao da Maratona de Nova York"
        )
        st.markdown(
            "Marilson Gomes dos Santos fez historia ao vencer a Maratona de Nova York em "
            "**2006** com o tempo de 2:09:58, e novamente em **2008** com 2:08:43. "
            "Ele se tornou o primeiro homem sul-americano a vencer a Maratona de Nova York, "
            "consolidando o lugar do Brasil na historia das maratonas mundiais. Suas vitorias "
            "permanecem entre as maiores conquistas de um atleta brasileiro na corrida de "
            "longa distancia."
        )

        st.markdown("---")
        st.markdown(
            "#### Ronaldo da Costa -- Ex-Recordista Mundial"
        )
        st.markdown(
            "Ronaldo da Costa estabeleceu o recorde mundial de maratona na **Maratona de "
            "Berlim de 1998** com o impressionante tempo de **2:06:05**, quebrando o recorde "
            "anterior em 45 segundos. Ele se tornou o primeiro corredor a completar uma "
            "maratona com parciais negativas em uma performance de recorde mundial, correndo "
            "a segunda metade mais rapido que a primeira. Sua conquista permanece como um "
            "marco historico para o atletismo brasileiro no cenario global."
        )

        st.markdown("---")
        st.markdown(
            "#### Crescimento da Representacao Brasileira"
        )
        st.markdown(
            "A participacao brasileira nas Maratonas Majors do Mundo cresceu de forma "
            "constante ao longo dos anos. A partir de um pequeno contingente de corredores "
            "dedicados, a presenca brasileira se expandiu para incluir centenas de "
            "finalistas em todas as seis grandes maratonas a cada ano. Esse crescimento "
            "reflete a popularidade crescente da corrida de longa distancia no Brasil e o "
            "comprometimento dos atletas brasileiros em competir no mais alto nivel do "
            "esporte. Os dados mostram uma melhoria consistente tanto nos numeros de "
            "participacao quanto nos tempos medios de desempenho, sinalizando um futuro "
            "promissor para a maratona brasileira."
        )


if __name__ == "__main__" or "streamlit" in os.path.basename(sys.argv[0]).lower():
    data_loaded = False
    results_df = None
    brazil_df = None

    if "data" in st.session_state and st.session_state["data"] is not None:
        data_dict = st.session_state["data"]
        results_df = data_dict.get("marathon_results")
        brazil_df = data_dict.get("brazil_analysis")
        if results_df is not None:
            data_loaded = True
    elif "filtered_data" in st.session_state and st.session_state["filtered_data"] is not None:
        results_df = st.session_state["filtered_data"]
        data_loaded = True

    if "marathon_results" in st.session_state and st.session_state["marathon_results"] is not None:
        results_df = st.session_state["marathon_results"]
        data_loaded = True
    elif "results" in st.session_state and st.session_state["results"] is not None:
        results_df = st.session_state["results"]
        data_loaded = True
    elif "combined_data" in st.session_state and st.session_state["combined_data"] is not None:
        results_df = st.session_state["combined_data"]
        data_loaded = True

    if "brazil_analysis" in st.session_state and st.session_state["brazil_analysis"] is not None:
        brazil_df = st.session_state["brazil_analysis"]

    if not data_loaded:
        try:
            from src.data.load_data import load_csv, get_data_filepaths
            filepaths = get_data_filepaths()
            raw_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw")
            processed_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data", "processed")

            combined_path = os.path.normpath(os.path.join(processed_dir, "combined_marathon_data.csv"))
            results_path = os.path.normpath(os.path.join(raw_dir, "marathon_results.csv"))
            brazil_path = os.path.normpath(os.path.join(processed_dir, "brazilian_runners_analysis.csv"))

            if os.path.exists(combined_path):
                results_df = pd.read_csv(combined_path)
                data_loaded = True
            elif os.path.exists(results_path):
                results_df = pd.read_csv(results_path)
                data_loaded = True

            if os.path.exists(brazil_path):
                brazil_df = pd.read_csv(brazil_path)
        except Exception:
            pass

    if data_loaded:
        render_brazil_page(results_df, brazil_df)
    else:
        st.warning(
            "Nenhum dado carregado. Verifique se os datasets de maratona estao disponiveis "
            "no diretorio de dados ou carregue-os pelo ponto de entrada principal do "
            "aplicativo para que sejam armazenados no estado da sessao."
        )
