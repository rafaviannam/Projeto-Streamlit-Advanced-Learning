"""
Painel de Monitoramento de Atrasos de Voos — ANAC
Projeto: Maria Isadora e Rafaela Vianna

Execute com:
    streamlit run app.py
"""

import json
import os
import warnings

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Configuração da página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Painel de Atrasos de Voos",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Funções auxiliares com cache
# ─────────────────────────────────────────────

@st.cache_resource
def carregar_modelo(nome_modelo: str):
    """Carrega o modelo .joblib uma única vez (fica em cache)."""
    caminho = os.path.join("models", f"{nome_modelo}_model.joblib")
    if not os.path.exists(caminho):
        return None
    return joblib.load(caminho)


@st.cache_data
def carregar_infos_modelo():
    """Carrega o JSON com nome do target e lista de features."""
    caminho = os.path.join("models", "model_infos.json")
    if not os.path.exists(caminho):
        return None
    with open(caminho) as f:
        return json.load(f)


@st.cache_data(ttl=600)
def carregar_dados(caminho_csv: str = "voos_clima.csv"):
    """Lê o CSV e faz limpeza mínima para exibição."""
    if not os.path.exists(caminho_csv):
        return None
    df = pd.read_csv(caminho_csv, low_memory=False)
    # Renomeia para o padrão do pipeline
    rename_map = {
        "partida_prevista":   "Partida.Prevista",
        "partida_real":       "Partida.Real",
        "chegada_prevista":   "Chegada.Prevista",
        "chegada_real":       "Chegada.Real",
        "situacao_voo":       "Situacao.Voo",
        "justificativa":      "Codigo.Justificativa",
        "icao_empresa_aerea": "Empresa",
        "numero_voo":         "Numero.Voo",
        "tipo_linha":         "Tipo.Linha",
        "icao_origem":        "Origem",
        "icao_destino":       "Destino",
        "codigo_autorizacao": "Codigo.Autorizacao",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Converte datas
    for col in ["Partida.Prevista", "Partida.Real"]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Cria coluna de atraso para análise exploratória
    if "Partida.Real" in df.columns and "Partida.Prevista" in df.columns:
        df["Atraso_min"] = (
            df["Partida.Real"] - df["Partida.Prevista"]
        ).dt.total_seconds() / 60
        df["Atrasado"] = (df["Atraso_min"] > 15).astype(int)

    return df


# ─────────────────────────────────────────────
# Carregamento dos recursos
# ─────────────────────────────────────────────

MODELOS_DISPONIVEIS = [
    "HistGradientBoostingClassifier",
    "LogisticRegression",
]

infos = carregar_infos_modelo()
df = carregar_dados()

# ─────────────────────────────────────────────
# Sidebar — controles globais
# ─────────────────────────────────────────────

st.sidebar.title("✈️ Painel ANAC")
st.sidebar.markdown("Monitoramento e previsão de atrasos de voos.")

st.sidebar.divider()
st.sidebar.subheader("Modelo")

modelo_selecionado = st.sidebar.selectbox(
    "Escolha o modelo:",
    MODELOS_DISPONIVEIS,
    index=0,
    help="Todos os modelos foram treinados no pipeline_treino() com dados de 2025.",
)

modelo = carregar_modelo(modelo_selecionado)

if modelo is None:
    st.sidebar.warning(
        f"⚠️ Arquivo `models/{modelo_selecionado}_model.joblib` não encontrado.\n\n"
        "Execute o `pipeline_treino()` do notebook para gerar os modelos."
    )

st.sidebar.divider()

# ─────────────────────────────────────────────
# Abas principais
# ─────────────────────────────────────────────

aba_prev, aba_monitor, aba_info = st.tabs(
    ["Prever Atraso", "Monitoramento da Base", "Sobre o Projeto"]
)

# ══════════════════════════════════════════════
# ABA 1 — PREVISÃO DE ATRASO
# ══════════════════════════════════════════════
with aba_prev:
    st.title("Previsão de Atraso de Voo")
    st.markdown(
        "Preencha os dados do voo abaixo. O modelo estimará a **probabilidade de atraso** (> 15 min)."
    )

    st.divider()

    # ── Formulário de entrada ────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        st.subheader("Voo")
        tipo_linha = st.selectbox(
            "Tipo de Linha",
            ["N - Nacional", "I - Internacional", "C - Cargueiro", "X - Outro"],
            index=0,
        )
        empresa = st.text_input("Empresa (ICAO)", value="GLO", max_chars=4).upper()
        origem = st.text_input("Origem (ICAO)", value="SBGR", max_chars=4).upper()
        destino = st.text_input("Destino (ICAO)", value="SBBR", max_chars=4).upper()

    with col2:
        st.subheader("Horário")
        hora_partida = st.slider("Hora prevista de partida", 0, 23, 8)
        dia_semana = st.selectbox(
            "Dia da semana",
            ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"],
            index=0,
        )
        mes = st.selectbox("Mês", list(range(1, 13)), index=0)
        fim_de_semana = 1 if dia_semana in ["Sábado", "Domingo"] else 0

    with col3:
        st.subheader("Clima (opcional)")
        precipitacao_origem = st.number_input("Precipitação origem (mm)", 0.0, 200.0, 0.0, step=0.5)
        umidade_origem = st.number_input("Umidade origem (%)", 0.0, 100.0, 70.0, step=1.0)
        precipitacao_destino = st.number_input("Precipitação destino (mm)", 0.0, 200.0, 0.0, step=0.5)
        umidade_destino = st.number_input("Umidade destino (%)", 0.0, 100.0, 70.0, step=1.0)

    st.divider()
    prever = st.button("Prever Atraso", type="primary", use_container_width=True)

    # ── Lógica de previsão ───────────────────────
    if prever:
        if modelo is None or infos is None:
            st.error(
                "Modelo ou `model_infos.json` não encontrados. "
                "Execute o `pipeline_treino()` no notebook primeiro."
            )
        else:
            tipo_codigo = tipo_linha.split(" - ")[0]
            dia_idx = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"].index(dia_semana)

            # Monta dicionário base com as variáveis que o modelo pode ter
            entrada_dict: dict = {
                "hora_partida": hora_partida,
                "Mes": mes,
                "DiaSemana": dia_idx,
                "FimDeSemana": fim_de_semana,
                "precipitacao_origem": precipitacao_origem,
                "umidade_origem": umidade_origem,
                "precipitacao_destino": precipitacao_destino,
                "umidade_destino": umidade_destino,
                # Frequency encoding (valores neutros — idealmente vêm do fit)
                "Origem": 0.5,
                "Destino": 0.5,
                "taxa_atraso_rota": 0.2,
                "taxa_atraso_empresa": 0.2,
                "taxa_atraso_hora": 0.2,
                "ocupacao_origem": 50.0,
            }

            # One-hot encoding para Empresa (gerado durante o data_treatment)
            colunas_modelo = infos.get("x_vars", [])
            for col in colunas_modelo:
                if col.startswith("Empresa_"):
                    cod = col.replace("Empresa_", "")
                    entrada_dict[col] = 1 if empresa == cod else 0
                if col.startswith("Tipo.Linha_"):
                    cod = col.replace("Tipo.Linha_", "")
                    entrada_dict[col] = 1 if tipo_codigo == cod else 0

            entrada = pd.DataFrame([entrada_dict])
            entrada = entrada.reindex(columns=colunas_modelo, fill_value=0).astype(np.float32)

            # Previsão
            try:
                if hasattr(modelo, "predict_proba"):
                    prob_atraso = float(modelo.predict_proba(entrada)[0][1])
                else:
                    prob_atraso = float(modelo.predict(entrada)[0])
            except Exception as e:
                st.error(f"Erro ao prever: {e}")
                prob_atraso = None

            if prob_atraso is not None:
                st.divider()
                c1, c2, c3 = st.columns([1, 2, 1])

                with c2:
                    st.metric(
                        label="Probabilidade de Atraso",
                        value=f"{prob_atraso:.1%}",
                        delta=f"Modelo: {modelo_selecionado.replace('Classifier', '')}",
                    )

                    # Gauge visual
                    fig_gauge = go.Figure(
                        go.Indicator(
                            mode="gauge+number",
                            value=prob_atraso * 100,
                            number={"suffix": "%", "font": {"size": 36}},
                            gauge={
                                "axis": {"range": [0, 100]},
                                "bar": {"color": "#e63946" if prob_atraso > 0.5 else "#2a9d8f"},
                                "steps": [
                                    {"range": [0, 30], "color": "#d8f3dc"},
                                    {"range": [30, 60], "color": "#fff3b0"},
                                    {"range": [60, 100], "color": "#ffccd5"},
                                ],
                                "threshold": {
                                    "line": {"color": "red", "width": 3},
                                    "thickness": 0.75,
                                    "value": 50,
                                },
                            },
                            title={"text": "Risco de Atraso"},
                        )
                    )
                    fig_gauge.update_layout(height=300, margin=dict(t=40, b=0))
                    st.plotly_chart(fig_gauge, use_container_width=True)

                    if prob_atraso >= 0.6:
                        st.error("⚠️ **Alto risco de atraso.** Considere acionar protocolos preventivos.")
                    elif prob_atraso >= 0.4:
                        st.warning("🟡 **Risco moderado.** Monitorar o voo com atenção.")
                    else:
                        st.success("✅ **Baixo risco de atraso.** Voo deve partir no prazo.")

                    st.caption(
                        "⚠️ A previsão é indicativa."
                    )


# ══════════════════════════════════════════════
# ABA 2 — MONITORAMENTO DA BASE
# ══════════════════════════════════════════════
with aba_monitor:
    st.title("Monitoramento da Base de Voos")

    if df is None:
        st.warning(
            "Arquivo `voos_clima.csv` não encontrado na pasta do projeto.\n\n"
            "Coloque o arquivo na mesma pasta que o `app.py` e recarregue a página."
        )
    else:
        # ── KPIs ────────────────────────────────────
        total_voos = len(df)
        pct_atraso = df["Atrasado"].mean() if "Atrasado" in df.columns else None
        n_empresas = df["Empresa"].nunique() if "Empresa" in df.columns else "—"
        n_rotas = (
            df[["Origem", "Destino"]].drop_duplicates().shape[0]
            if "Origem" in df.columns
            else "—"
        )

        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Total de Voos", f"{total_voos:,}")
        k2.metric(
            "Taxa de Atraso",
            f"{pct_atraso:.1%}" if pct_atraso is not None else "—",
        )
        k3.metric("Empresas", n_empresas)
        k4.metric("Rotas Únicas", f"{n_rotas:,}" if isinstance(n_rotas, int) else n_rotas)

        st.divider()

        # ── Filtros ──────────────────────────────────
        st.subheader("Filtros")
        fcol1, fcol2, fcol3 = st.columns(3)

        empresas_lista = ["Todas"] + (
            sorted(df["Empresa"].dropna().unique().tolist())
            if "Empresa" in df.columns
            else []
        )
        empresa_filtro = fcol1.selectbox("Empresa", empresas_lista)

        tipos_lista = ["Todos"] + (
            sorted(df["Tipo.Linha"].dropna().unique().tolist())
            if "Tipo.Linha" in df.columns
            else []
        )
        tipo_filtro = fcol2.selectbox("Tipo de Linha", tipos_lista)

        situacoes = ["Todas"] + (
            sorted(df["Situacao.Voo"].dropna().unique().tolist())
            if "Situacao.Voo" in df.columns
            else []
        )
        situacao_filtro = fcol3.selectbox("Situação do Voo", situacoes)

        # Aplica filtros
        df_f = df.copy()
        if empresa_filtro != "Todas" and "Empresa" in df_f.columns:
            df_f = df_f[df_f["Empresa"] == empresa_filtro]
        if tipo_filtro != "Todos" and "Tipo.Linha" in df_f.columns:
            df_f = df_f[df_f["Tipo.Linha"] == tipo_filtro]
        if situacao_filtro != "Todas" and "Situacao.Voo" in df_f.columns:
            df_f = df_f[df_f["Situacao.Voo"] == situacao_filtro]

        st.caption(f"Exibindo **{len(df_f):,}** voos após filtros.")
        st.divider()

        # ── Gráficos ─────────────────────────────────
        g1, g2 = st.columns(2)

        with g1:
            st.subheader("Atrasos por Empresa")
            if "Empresa" in df_f.columns and "Atrasado" in df_f.columns:
                emp_atraso = (
                    df_f.groupby("Empresa")["Atrasado"]
                    .agg(["mean", "sum", "count"])
                    .reset_index()
                    .rename(columns={"mean": "taxa", "sum": "atrasados", "count": "total"})
                    .sort_values("taxa", ascending=False)
                    .head(15)
                )
                fig = px.bar(
                    emp_atraso,
                    x="Empresa",
                    y="taxa",
                    labels={"taxa": "Taxa de Atraso", "Empresa": "Empresa"},
                    color="taxa",
                    color_continuous_scale="RdYlGn_r",
                    text=emp_atraso["taxa"].apply(lambda x: f"{x:.0%}"),
                )
                fig.update_layout(coloraxis_showscale=False, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Dados insuficientes para este gráfico.")

        with g2:
            st.subheader("Distribuição de Atrasos (minutos)")
            if "Atraso_min" in df_f.columns:
                atraso_clip = df_f["Atraso_min"].clip(-30, 120).dropna()
                fig2 = px.histogram(
                    atraso_clip,
                    nbins=60,
                    labels={"value": "Minutos de Atraso", "count": "Quantidade"},
                    color_discrete_sequence=["#457b9d"],
                )
                fig2.add_vline(x=15, line_dash="dash", line_color="red", annotation_text="15 min")
                fig2.add_vline(x=0, line_dash="dot", line_color="gray")
                st.plotly_chart(fig2, use_container_width=True)
            else:
                st.info("Coluna de atraso não disponível.")

        g3, g4 = st.columns(2)

        with g3:
            st.subheader("Atrasos por Hora do Dia")
            if "Partida.Prevista" in df_f.columns and "Atrasado" in df_f.columns:
                df_hora = df_f.copy()
                df_hora["hora"] = df_hora["Partida.Prevista"].dt.hour
                hora_atraso = (
                    df_hora.groupby("hora")["Atrasado"].mean().reset_index()
                )
                fig3 = px.line(
                    hora_atraso,
                    x="hora",
                    y="Atrasado",
                    markers=True,
                    labels={"hora": "Hora do Dia", "Atrasado": "Taxa de Atraso"},
                    color_discrete_sequence=["#e63946"],
                )
                fig3.update_yaxes(tickformat=".0%")
                st.plotly_chart(fig3, use_container_width=True)
            else:
                st.info("Dados de horário não disponíveis.")

        with g4:
            st.subheader("Top 10 Rotas com Mais Atrasos")
            if "Origem" in df_f.columns and "Destino" in df_f.columns and "Atrasado" in df_f.columns:
                df_rota = df_f.copy()
                df_rota["Rota"] = df_rota["Origem"] + " → " + df_rota["Destino"]
                rota_atraso = (
                    df_rota.groupby("Rota")["Atrasado"]
                    .agg(["mean", "count"])
                    .reset_index()
                    .query("count >= 30")  # mínimo de voos para ser relevante
                    .sort_values("mean", ascending=False)
                    .head(10)
                    .rename(columns={"mean": "taxa_atraso", "count": "total_voos"})
                )
                fig4 = px.bar(
                    rota_atraso,
                    x="taxa_atraso",
                    y="Rota",
                    orientation="h",
                    labels={"taxa_atraso": "Taxa de Atraso", "Rota": ""},
                    color="taxa_atraso",
                    color_continuous_scale="RdYlGn_r",
                    text=rota_atraso["taxa_atraso"].apply(lambda x: f"{x:.0%}"),
                )
                fig4.update_layout(
                    yaxis={"categoryorder": "total ascending"},
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig4, use_container_width=True)
            else:
                st.info("Dados de rota não disponíveis.")

        # ── Tabela ───────────────────────────────────
        st.subheader("Dados Brutos (amostra)")
        cols_exibir = [
            c for c in [
                "Empresa", "Tipo.Linha", "Origem", "Destino",
                "Partida.Prevista", "Situacao.Voo", "Atraso_min", "Atrasado",
            ]
            if c in df_f.columns
        ]
        st.dataframe(
            df_f[cols_exibir].head(500),
            use_container_width=True,
            height=300,
        )


# ══════════════════════════════════════════════
# ABA 3 — SOBRE O PROJETO
# ══════════════════════════════════════════════
with aba_info:
    st.title("Sobre o Projeto")

    st.markdown("""
    ### Modelo Preditivo de Atrasos de Voos — ANAC

    **Autoras:** Maria Isadora Camargo Davida · Rafaela Vianna  
    **Instituição:** Fundação Getulio Vargas — Escola de Relações Internacionais  
    **Disciplina:** Advanced Learning — 2026.1 · Prof. Ronan Cunha

    ---

    ### Problema de Negócio

    O setor de aviação civil no Brasil é regulado pela ANAC (Resolução n° 400).
    Atrasos geram custos operacionais extras e possível indenização judicial por danos morais.
    As companhias têm capacidade limitada de prever esses eventos com antecedência.

    ### Solução

    Um modelo preditivo que estima a **probabilidade de atraso** de um voo antes da partida,
    apoiando decisões operacionais de forma proativa.

    ### Modelos Treinados

    | Modelo | AUC-ROC | F1 | Acurácia |
    |---|---|---|---|
    | Random Forest | 0,5010 | 0,7158 | 0,7183 |
    | Árvore de Decisão | 0,5015 | 0,5749 | 0,5160 |
    | Hist Gradient Boosting | 0,5109 | 0,5016 | 0,4441 |
    | Regressão Logística | 0,4462 | 0,7495 | 0,8276 |

    > **Nota:** Todos os modelos apresentaram AUC-ROC próximo de 0,5, indicando
    > dificuldade em discriminar as classes. A hipótese principal é o desbalanceamento
    > do target (83,9% "no prazo" vs 16,1% "atrasado").

    ### Stakeholders

    Companhias aéreas · Passageiros · Aeroportos · Órgãos reguladores (ANAC)
    """)

    st.divider()
    st.caption("Painel desenvolvido com Streamlit · Dados: ANAC + INMET")
