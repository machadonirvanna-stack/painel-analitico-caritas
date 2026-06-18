import re
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Registro de Assessoramento ATI", layout="wide")

# ==================================================================================
# 1. MAPEAMENTO DE COLUNAS E DICIONÁRIOS GLOBAIS
# ==================================================================================
COLS = {
    "tipo": "Qual é o tipo de agenda que você deseja registrar?",
    "data": "Data",
    "assessor1": "Assessor 1",
    "assessor2": "Assessor 2",
    "demais": "Demais agentes Cáritas",
    "caso_sensivel": "Caso sensível",
    "formato": "Formato do assessoramento",
    "local": "Local do assessoramento",
    "inicio": "Hora de início",
    "fim": "Hora de fim",
    "doc": "Algum documento foi produzido durante o assessoramento?",
    "drive": "A documentação foi salva no drive compartilhado da ATI?",
    "encaminhamentos": "Encaminhamentos",
    "relatorio": "Relatório",
    "comunidade": "Comunidade(s) acompanhada(s) no Assessoramento Coletivo.",
    "modalidade_coletivo": "Qual é a modalidade do Assessoramento Coletivo?",
    "assunto_coletivo": "Assunto do Assessoramento Coletivo",
    "assunto_demanda": "Qual é o assunto do Assessoramento de demanda espontânea?",
    "assunto_repactuacao": "Qual é o assunto relacionado à repactuação?",
    "assunto_reassentamento": "Qual é o assunto do Assessoramento sobre o Reassentamento?",
    "problemas_reassentamento": "Indique quais os problemas você identificou no atendimento feito no reassentamento.",
    "modalidade_reassentamento": "Esse atendimento se refers a qual modalidade de reassentamento?",
    "feminino": "Quantas pessoas do sexo feminino?",
    "masculino": "Quantas pessoas do sexo masculino?",
    "criancas": "Quantas crianças participaram do atendimento?",
    "adolescentes": "Quantos adolescentes participaram do atendimento?",
    "jovens": "Quantos Jovens participaram?",
    "adultos": "Quantos Adultos participaram?",
    "idosos": "Quantos Idosos participaram?",
    "samarco": "Essa demanda se trata de um acompanhamento de atendimento junto à Samarco?",
}

COL_ALIASES = {
    "assessor2": ["Assessor 2", "Assessor "],
    "assunto_coletivo": ["Assunto do Assessoramento Coletivo", "Qual é o assunto do Assessoramento Coletivo?"],
    "assunto_demanda": [
        "Qual é o assunto do Assessoramento de demanda espontânea?",
        'Qual é o assunto do Assessoramento de demanda espontânea?"'
    ],
    "assunto_repactuacao": ["Qual é o assunto relacionado à repactuação?"],
    "assunto_reassentamento": [
        "Qual é o assunto do Assessoramento sobre o Reassentamento?",
        "Assunto do Assessoramento Individual Reassentamento - Qual é o enquadramento?",
        "Qual o enquadramento?",
        "Qual é o assunto do Assessoramento Individual?"
    ],
    "modalidade_reassentamento": [
        "Esse atendimento se refere a qual modalidade de reassentamento?",
        "Esse atendimento se refers a qual modalidade de reassentamento?"
    ],
    "comunidade": ["Comunidade(s) acompanhada(s) no Assessoramento Coletivo."],
    "meio_contato": ["Meio de contato"],
    "observacao": ["Observação"],

    "id_individual": [
        "Indique o ID do assessoramento",
        "Assessoramento Individual - ID"
    ],
    "id_coletivo": [
        "Indique o ID do assessoramento 2",
        "Assessoramento Coletivo - ID"
    ],
    "id_parceiros": [
        "Assessoramento Parceiros - ID"
    ],
    "codigo_nf": [
        "Inserir o Código do Núcleo familiar.",
        "Demanda Espontânea - Código do NF",
        "Assessoramento com ID - Código do NF"
    ],
}

PARTICIPANTES = ["feminino", "masculino", "criancas", "adolescentes", "jovens", "adultos", "idosos"]
STOPWORDS_PT = ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "na", "no", "uma", "os", "as", "dos", "das", "ao", "aos", "por", "mais", "se", "foi", "atendimento", "reunião", "assessoramento", "comunidade", "relatório", "visita", "sobre", "nao", "não", "como"]

# ==================================================================================
# 2. FUNÇÕES DE AUXÍLIO / TRATAMENTO DE DADOS
# ==================================================================================
def read_sheet_safely(file) -> pd.DataFrame:
    if str(file).endswith(".csv"):
        return pd.read_csv(file)

    preview = pd.read_excel(file, header=None, nrows=10)

    header_row = 0

    for i in range(len(preview)):
        row_values = [str(v).strip() for v in preview.iloc[i].tolist()]

        tem_data = any(v == "Data" for v in row_values)
        tem_tipo = any("tipo de agenda" in v.lower() for v in row_values)

        if tem_data and tem_tipo:
            header_row = i
            break

    return pd.read_excel(file, header=header_row)


def coalesce_columns(df: pd.DataFrame, candidates: list) -> pd.Series:
    existentes = [c for c in candidates if c in df.columns]

    if not existentes:
        return pd.Series(pd.NA, index=df.index)

    resultado = df[existentes[0]]

    for coluna in existentes[1:]:
        resultado = resultado.combine_first(df[coluna])

    return resultado


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).replace("_x000a_", "\n").strip() for c in df.columns]

    df = df.rename(columns={v: k for k, v in COLS.items() if v in df.columns})

    for key, aliases in COL_ALIASES.items():
        if key not in df.columns:
            df[key] = coalesce_columns(df, aliases)

    return df

def to_number(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s.astype(str).str.extract(r"(\d+(?:[,.]\d+)?)", expand=False).str.replace(",", ".", regex=False), errors="coerce").fillna(0)

def parse_time(value):
    if pd.isna(value): return pd.NaT
    if hasattr(value, "hour") and hasattr(value, "minute"):
        return pd.Timestamp(year=2000, month=1, day=1, hour=value.hour, minute=value.minute)
    text = str(value).strip()
    match = re.search(r"(\d{1,2}):(\d{2})", text)
    if not match: return pd.NaT
    return pd.Timestamp(year=2000, month=1, day=1, hour=int(match.group(1)), minute=int(match.group(2)))

def split_multi_values(series: pd.Series) -> pd.Series:
    values = []
    for item in series.dropna().astype(str):
        parts = re.split(r",|;|\n| e ", item)
        values.extend([p.strip() for p in parts if p.strip() and p.strip().lower() not in ["não se aplica", "nan", "não"]])
    return pd.Series(values, dtype="string")

def extrair_palavras_chave(series: pd.Series, top_n=15):
    texto_completo = " ".join(series.dropna().astype(str).str.lower())
    palavras = re.findall(r'\b[a-záéíóúçãõâêô]{4,}\b', texto_completo)
    filtradas = [p for p in palavras if p not in STOPWORDS_PT]
    if not filtradas:
        return pd.DataFrame(columns=["Termo", "Frequência"])
    return pd.Series(filtradas).value_counts().reset_index(name="Frequência").rename(columns={"index": "Termo"}).head(top_n)

def formatar_variacao(valor):
    if pd.isna(valor):
        return "Sem base anterior"
    if valor > 0:
        return f"+{valor:.1f}%"
    return f"{valor:.1f}%"


def calcular_variacao_periodo(df_base, coluna_data="data", meses_janela=1):
    if df_base.empty or coluna_data not in df_base.columns:
        return {
            "atual": 0,
            "anterior": 0,
            "variacao": np.nan,
            "texto": "Sem dados suficientes"
        }

    temp = df_base.copy()
    temp[coluna_data] = pd.to_datetime(temp[coluna_data], errors="coerce")
    temp = temp[temp[coluna_data].notna()].copy()

    if temp.empty:
        return {
            "atual": 0,
            "anterior": 0,
            "variacao": np.nan,
            "texto": "Sem dados suficientes"
        }

    ultimo_mes = temp[coluna_data].max().to_period("M")

    meses_atual = [
        (ultimo_mes - i).strftime("%Y-%m")
        for i in range(meses_janela)
    ]

    meses_anterior = [
        (ultimo_mes - meses_janela - i).strftime("%Y-%m")
        for i in range(meses_janela)
    ]

    temp["ano_mes_calc"] = temp[coluna_data].dt.to_period("M").astype(str)

    atual = temp[temp["ano_mes_calc"].isin(meses_atual)].shape[0]
    anterior = temp[temp["ano_mes_calc"].isin(meses_anterior)].shape[0]

    if anterior == 0:
        variacao = np.nan
        texto = "Sem base anterior"
    else:
        variacao = ((atual - anterior) / anterior) * 100
        texto = formatar_variacao(variacao)

    return {
        "atual": atual,
        "anterior": anterior,
        "variacao": variacao,
        "texto": texto
    }


def calcular_variacao_indicador(df_base, filtro_coluna=None, filtro_valor=None, meses_janela=1):
    if df_base.empty:
        return {
            "atual": 0,
            "anterior": 0,
            "variacao": np.nan,
            "texto": "Sem dados suficientes"
        }

    temp = df_base.copy()

    if filtro_coluna and filtro_coluna in temp.columns:
        temp = temp[temp[filtro_coluna] == filtro_valor]

    return calcular_variacao_periodo(temp, meses_janela=meses_janela)


def top_riscos_emergentes(df_base, coluna_tema="tema_principal", meses_janela=3, top_n=10):
    if df_base.empty:
        return pd.DataFrame()

    if "data" not in df_base.columns or coluna_tema not in df_base.columns:
        return pd.DataFrame()

    temp = df_base.copy()
    temp["data"] = pd.to_datetime(temp["data"], errors="coerce")
    temp = temp[temp["data"].notna()].copy()
    temp = temp[temp[coluna_tema].notna()].copy()

    if temp.empty:
        return pd.DataFrame()

    ultimo_mes = temp["data"].max().to_period("M")

    meses_atual = [
        (ultimo_mes - i).strftime("%Y-%m")
        for i in range(meses_janela)
    ]

    meses_anterior = [
        (ultimo_mes - meses_janela - i).strftime("%Y-%m")
        for i in range(meses_janela)
    ]

    temp["ano_mes_calc"] = temp["data"].dt.to_period("M").astype(str)

    atual = (
        temp[temp["ano_mes_calc"].isin(meses_atual)]
        .groupby(coluna_tema)
        .size()
        .reset_index(name="periodo_atual")
    )

    anterior = (
        temp[temp["ano_mes_calc"].isin(meses_anterior)]
        .groupby(coluna_tema)
        .size()
        .reset_index(name="periodo_anterior")
    )

    comparativo = atual.merge(anterior, on=coluna_tema, how="left")
    comparativo["periodo_anterior"] = comparativo["periodo_anterior"].fillna(0)

    comparativo["crescimento_abs"] = (
        comparativo["periodo_atual"] - comparativo["periodo_anterior"]
    )

    comparativo["crescimento_pct"] = np.where(
        comparativo["periodo_anterior"] > 0,
        (
            (comparativo["periodo_atual"] - comparativo["periodo_anterior"])
            / comparativo["periodo_anterior"]
        ) * 100,
        np.nan
    )

    comparativo = comparativo[
        (comparativo["periodo_atual"] >= 2) &
        (comparativo["crescimento_abs"] > 0)
    ]

    if comparativo.empty:
        return pd.DataFrame()

    return comparativo.sort_values(
        ["crescimento_pct", "crescimento_abs"],
        ascending=False,
        na_position="last"
    ).head(top_n)


def gerar_alertas_estrategicos(df_base):
    alertas = []

    if df_base.empty:
        return alertas

    df_base = df_base.copy()

    if "data" in df_base.columns:
        df_base["data"] = pd.to_datetime(df_base["data"], errors="coerce")

    if "caso_sensivel_flag" in df_base.columns:
        var_sensiveis = calcular_variacao_indicador(
            df_base,
            filtro_coluna="caso_sensivel_flag",
            filtro_valor=True,
            meses_janela=1
        )

        if not pd.isna(var_sensiveis["variacao"]) and var_sensiveis["variacao"] >= 30:
            alertas.append(
                f"Casos sensíveis aumentaram {var_sensiveis['texto']} em relação ao mês anterior."
            )

    var_atendimentos_3m = calcular_variacao_periodo(df_base, meses_janela=3)

    if not pd.isna(var_atendimentos_3m["variacao"]) and var_atendimentos_3m["variacao"] >= 25:
        alertas.append(
            f"O volume de atendimentos dos últimos 3 meses cresceu {var_atendimentos_3m['texto']} frente ao trimestre anterior."
        )

    colunas_pendencia = [
        col for col in ["doc_flag", "drive_flag", "tem_relatorio"]
        if col in df_base.columns
    ]

    if colunas_pendencia:
        pendencias = df_base[
            df_base[colunas_pendencia].eq(False).any(axis=1)
        ]

        perc_pendencias = (
            pendencias.shape[0] / df_base.shape[0] * 100
            if df_base.shape[0] > 0
            else 0
        )

        if perc_pendencias >= 30:
            alertas.append(
                f"{perc_pendencias:.1f}% dos registros possuem alguma pendência de evidência, relatório ou drive."
            )

    comunidade_col = None

    if "comunidade_analise" in df_base.columns:
        comunidade_col = "comunidade_analise"
    elif "local" in df_base.columns:
        comunidade_col = "local"

    if comunidade_col and "data" in df_base.columns:
        temp_local = df_base[df_base["data"].notna()].copy()

        if not temp_local.empty:
            ultima_data_por_local = (
                temp_local.groupby(comunidade_col)["data"]
                .max()
                .reset_index()
            )

            data_referencia = temp_local["data"].max()

            ultima_data_por_local["dias_sem_atendimento"] = (
                data_referencia - ultima_data_por_local["data"]
            ).dt.days

            locais_sem_atendimento = ultima_data_por_local[
                ultima_data_por_local["dias_sem_atendimento"] >= 60
            ].sort_values("dias_sem_atendimento", ascending=False)

            for _, row in locais_sem_atendimento.head(5).iterrows():
                alertas.append(
                    f"A comunidade {row[comunidade_col]} está há {int(row['dias_sem_atendimento'])} dias sem atendimento registrado."
                )

    if "assessor1" in df_base.columns and "ind_sobrecarga" in df_base.columns:
        carga = df_base.groupby("assessor1")["ind_sobrecarga"].sum()
        media_carga = carga.mean()

        if media_carga > 0:
            sobrecarregados = carga[
                carga > media_carga * 1.5
            ].sort_values(ascending=False)

            for assessor, valor in sobrecarregados.head(3).items():
                alertas.append(
                    f"O assessor {assessor} apresenta carga acumulada acima de 50% da média da equipe."
                )

    return alertas


def gerar_resumo_executivo(df_base):
    if df_base.empty:
        return "Não há dados disponíveis para gerar o resumo executivo."

    total_atendimentos = df_base.shape[0]
    total_participantes = int(df_base["total_participantes"].sum()) if "total_participantes" in df_base.columns else 0
    total_sensiveis = int(df_base["caso_sensivel_flag"].sum()) if "caso_sensivel_flag" in df_base.columns else 0
    total_encaminhamentos = int(df_base["tem_encaminhamento"].sum()) if "tem_encaminhamento" in df_base.columns else 0

    comunidade_col = "comunidade_analise" if "comunidade_analise" in df_base.columns else "local"

    comunidade_top = (
        df_base[comunidade_col].value_counts().idxmax()
        if comunidade_col in df_base.columns and not df_base[comunidade_col].dropna().empty
        else "não informada"
    )

    tema_top = (
        df_base["tema_principal"].value_counts().idxmax()
        if "tema_principal" in df_base.columns and not df_base["tema_principal"].dropna().empty
        else "não informado"
    )

    risco_top = (
        df_base.groupby(comunidade_col)["score_risco_territorial"]
        .mean()
        .sort_values(ascending=False)
        .index[0]
        if comunidade_col in df_base.columns and "score_risco_territorial" in df_base.columns
        else "não informado"
    )

    qualidade_media = df_base["ind_qualidade"].mean() if "ind_qualidade" in df_base.columns else 0

    variacao_mes = calcular_variacao_periodo(df_base, meses_janela=1)
    variacao_tri = calcular_variacao_periodo(df_base, meses_janela=3)
    variacao_sem = calcular_variacao_periodo(df_base, meses_janela=6)

    inicio = df_base["data"].min().strftime("%d/%m/%Y")
    fim = df_base["data"].max().strftime("%d/%m/%Y")

    alertas = gerar_alertas_estrategicos(df_base)

    riscos_emergentes = top_riscos_emergentes(df_base, meses_janela=3, top_n=3)

    if not riscos_emergentes.empty:
        lista_riscos = []
        for _, row in riscos_emergentes.iterrows():
            crescimento = "novo ou sem base anterior" if pd.isna(row["crescimento_pct"]) else f"{row['crescimento_pct']:.1f}%"
            lista_riscos.append(
                f"{row['tema_principal']} com crescimento de {crescimento}"
            )
        texto_riscos = "; ".join(lista_riscos)
    else:
        texto_riscos = "não foram identificados riscos emergentes com base suficiente no período filtrado"

    if alertas:
        texto_alertas = "\n".join([f"- {a}" for a in alertas[:5]])
    else:
        texto_alertas = "- Nenhum alerta crítico identificado nos filtros aplicados."

    texto = f"""
Resumo executivo do período de {inicio} a {fim}

Foram registrados {total_atendimentos} atendimentos, alcançando {total_participantes} participantes.

A comunidade com maior volume de atendimento foi {comunidade_top}. O tema mais frequente foi {tema_top}.

Foram identificados {total_sensiveis} casos sensíveis e {total_encaminhamentos} registros com encaminhamento.

A qualidade média das evidências ficou em {qualidade_media:.1f}%.

Na comparação com o mês anterior, o volume de atendimentos apresentou variação de {variacao_mes['texto']}. Considerando os últimos 3 meses frente ao trimestre anterior, a variação foi de {variacao_tri['texto']}. Considerando os últimos 6 meses frente ao semestre anterior, a variação foi de {variacao_sem['texto']}.

O território com maior score médio de risco foi {risco_top}.

Entre os riscos emergentes, {texto_riscos}.

Alertas de gestão:
{texto_alertas}

Leitura geral:
O painel permite acompanhar volume, território, temas, vulnerabilidades, reincidência, equipe, qualidade das evidências e tendência de demanda. As novas análises fortalecem a gestão porque mostram não apenas o que aconteceu, mas também o que está aumentando, onde há maior risco, onde há baixa cobertura e quais pontos exigem atenção da coordenação.
"""

    return texto.strip()

# ==================================================================================
# 3. CARREGAMENTO E ENGENHARIA DE ATRIBUTOS (INDICADORES DERIVADOS)
# ==================================================================================
@st.cache_data(show_spinner=False)
def interpretar(texto):
    st.caption(f"Leitura: {texto}")


def grafico_ranking_horizontal(df, x, y, titulo, texto=None, top_n=10):
    dados = df.sort_values(x, ascending=False).head(top_n).copy()

    fig = px.bar(
    df,
    x=x,
    y=y,
    orientation="h",
    text=df[x].round(1)
    )

    fig.update_layout(
        height=420,
        margin=dict(l=10, r=10, t=60, b=10)
    )

    st.plotly_chart(fig, use_container_width=True)

    if texto:
        interpretar(texto)
@st.cache_data(show_spinner="Carregando e preparando os dados...")
def load_data(file) -> pd.DataFrame:
    df = read_sheet_safely(file)
    df = normalize_columns(df)
    
    colunas_padrao = {
        "tipo": "Não informado", "assessor1": "Não informado", "assessor2": "",
        "local": "Não informado", "formato": "Não informado", "encaminhamentos": "",
        "relatorio": "", "caso_sensivel": "", "doc": "", "drive": "", "modalidade_reassentamento": "",
        "samarco": "Não preenchido", "demais": ""
    }

    for coluna, valor in colunas_padrao.items():
        if coluna not in df.columns: df[coluna] = valor
    
    df["data"] = pd.to_datetime(df["data"], errors="coerce")
    df = df[df["data"].notna()].copy()
    
    df["ano"] = df["data"].dt.year
    df["mes"] = df["data"].dt.month
    df["ano_mes"] = df["data"].dt.to_period("M").astype(str)
    id_cols = ["id_individual", "id_coletivo", "id_parceiros", "codigo_nf"]

    for coluna in id_cols:
        if coluna not in df.columns:
            df[coluna] = ""

    df["id_caso"] = (
        df[id_cols]
        .replace(["", "nan", "NaN", "Não informado", "Não se aplica"], pd.NA)
        .bfill(axis=1)
        .iloc[:, 0]
    )

    df["id_caso"] = df["id_caso"].fillna("").astype(str).str.strip()

    df["tem_id_caso"] = df["id_caso"].str.len() > 3
    dias = {"Monday": "Segunda-feira", "Tuesday": "Terça-feira", "Wednesday": "Quarta-feira", "Thursday": "Quinta-feira", "Friday": "Sexta-feira", "Saturday": "Sábado", "Sunday": "Domingo"}
    df["dia_semana"] = df["data"].dt.day_name().map(dias)
    
    for key in PARTICIPANTES:
        if key in df.columns: df[key] = to_number(df[key])
    df["total_participantes"] = df[[c for c in PARTICIPANTES if c in df.columns]].sum(axis=1)

    inicio = df["inicio"].map(parse_time) if "inicio" in df else pd.Series(pd.NaT, index=df.index)
    fim = df["fim"].map(parse_time) if "fim" in df else pd.Series(pd.NaT, index=df.index)
    df["duracao_horas"] = (fim - inicio).dt.total_seconds() / 3600
    df.loc[df["duracao_horas"] < 0, "duracao_horas"] += 24
    df["duracao_horas"] = df["duracao_horas"].clip(lower=0.1, upper=12).fillna(1.0)

    assuntos = [c for c in ["assunto_demanda", "assunto_repactuacao", "assunto_reassentamento", "assunto_coletivo"] if c in df.columns]
    df["tema_principal"] = df[assuntos].bfill(axis=1).iloc[:, 0] if assuntos else "Não informado"
    df["tema_principal"] = df["tema_principal"].fillna("Não informado").astype(str).str.strip()
    df["local"] = df.get("local", "Não Informado").fillna("Não Informado").astype(str).str.strip()
    df["formato"] = df.get("formato", "Não Informado").fillna("Não Informado").astype(str).str.strip()
    df["comunidade_analise"] = df.get("comunidade", "").fillna("").astype(str).str.strip()
    df.loc[
        df["comunidade_analise"].isin(["", "nan", "NaN", "Não informado"]),
        "comunidade_analise"
    ] = df["local"]

    df["genero_predominante"] = np.select(
        [
            df.get("feminino", 0) > df.get("masculino", 0),
            df.get("masculino", 0) > df.get("feminino", 0),
            (df.get("feminino", 0) > 0) & (df.get("masculino", 0) > 0)
        ],
        ["Feminino", "Masculino", "Misto"],
        default="Não informado"
)
    df["tem_encaminhamento"] = df.get("encaminhamentos", "").fillna("").astype(str).str.len() > 5
    df["tem_relatorio"] = df.get("relatorio", "").fillna("").astype(str).str.len() > 20
    df["caso_sensivel_flag"] = df.get("caso_sensivel", "").fillna("").astype(str).str.lower().str.contains("sim")
    df["doc_flag"] = df.get("doc", "").fillna("").astype(str).str.lower().str.contains("sim")
    df["drive_flag"] = df.get("drive", "").fillna("").astype(str).str.lower().str.contains("sim")
    df["is_reassentamento"] = df.get("modalidade_reassentamento", "").fillna("").astype(str).str.len() > 3

    tema_counts = df["tema_principal"].value_counts().to_dict()
    comunidade_counts = df["local"].value_counts().to_dict()
    df["freq_tema"] = df["tema_principal"].map(tema_counts)
    df["freq_comunidade"] = df["local"].map(comunidade_counts)

    # --- CÁLCULO DOS ÍNDICES ORIGINAIS ---
    df["ind_complexidade"] = ((3 * df["caso_sensivel_flag"].astype(int)) + (2 * df["tem_encaminhamento"].astype(int)) + (2 * df["doc_flag"].astype(int)) + (2 * df["is_reassentamento"].astype(int)) + (1 * df["duracao_horas"]))
    df["ind_sobrecarga"] = df["duracao_horas"] + df["ind_complexidade"] + (df["caso_sensivel_flag"].astype(int) * 5)
    df["ind_reincidencia"] = df["freq_tema"] + df["freq_comunidade"]
    df["ind_vulnerabilidade"] = (df["caso_sensivel_flag"].astype(int) * 5) + df["ind_complexidade"] + df["ind_reincidencia"]
    df["ind_qualidade"] = (df["doc_flag"].astype(int) + df["tem_relatorio"].astype(int) + df["drive_flag"].astype(int)) / 3 * 100
    df["ind_eficiencia"] = df["total_participantes"] / df["duracao_horas"].clip(lower=0.1)
    
    # --- [INSERÇÃO]: SCORE ÚNICO DE RISCO TERRITORIAL COMBINADO ---
    raw_score = df["ind_vulnerabilidade"] + (df["ind_reincidencia"] * 1.5) + (df["caso_sensivel_flag"].astype(int) * 10)
    amplitude = raw_score.max() - raw_score.min()
    if amplitude == 0:
        df["score_risco_territorial"] = 0.0
    else:
        df["score_risco_territorial"] = (((raw_score - raw_score.min()) / amplitude) * 100).round(1)

    # --- [INSERÇÃO]: DESMEMBRAMENTO DE TIPOLOGIA DE DEMANDA ---
    def classificar_demanda(row):
        tipo_agenda = str(row.get("tipo", "")).lower()
        if "coletivo" in tipo_agenda or "reunião" in tipo_agenda:
            return "Coletiva"
        elif str(row.get("samarco", "")).strip().lower() == "sim" or (pd.notna(row.get("demais")) and str(row.get("demais")).strip() != ""):
            return "Com Parceiros"
        elif ("espontânea" in tipo_agenda or str(row.get("assunto_demanda", "")).strip().lower() not in ["", "nan", "não informado"]):
            return "Individual (Espontânea)"
        else:
            return "Individual (Geral)"
    df["classificacao_demanda"] = df.apply(classificar_demanda, axis=1)

    # --- [INSERÇÃO]: MEIOS E CANAIS REMOTOS DETALHADOS ---
    def detalhar_remoto(row):
        if str(row.get("formato", "")).lower() == "remoto":
            txt_contexto = (str(row.get("relatorio", "")) + " " + str(row.get("encaminhamentos", ""))).lower()
            if "whatsapp" in txt_contexto or "zap" in txt_contexto: return "WhatsApp"
            elif "teams" in txt_contexto or "online" in txt_contexto or "virtual" in txt_contexto: return "Teams / Online"
            elif "telefone" in txt_contexto or "ligação" in txt_contexto: return "Telefone"
            else: return "Remoto (Outros/Não especificado)"
        return "Presencial"
    df["detalhe_remoto"] = df.apply(detalhar_remoto, axis=1)

    # --- [INSERÇÃO]: FILTRO E TRACKING DE ENCAMINHAMENTOS ---
    df["foi_encaminhado"] = df["tem_encaminhamento"].apply(lambda x: "Encaminhado" if x else "Resolvido Internamente")

    return df

@st.cache_data(show_spinner=False)
def preparar_opcoes_filtros(df):
    return {
        "anos": sorted(df["ano"].dropna().unique()),
        "tipos": sorted(df["tipo"].dropna().unique()),
        "assuntos": sorted(df["tema_principal"].dropna().unique()),
        "comunidades": sorted(df["comunidade_analise"].dropna().unique())
    }
# ==================================================================================
# 4. INTERFACE E MENU LATERAL
# ==================================================================================
# --- CARREGAMENTO AUTOMÁTICO E SEGURO DA BASE OFICIAL ---
bases = [
    Path("02. Registro de Assessoramento - antigo  (respostas).xlsx"),
    Path("02. Registro de Assessoramento - Atualizado  (respostas).xlsx")
]

dfs = []

for base in bases:
    if base.exists():
        temp = load_data(base)
        temp["base_origem"] = "Antiga" if "antigo" in base.name.lower() else "Atualizada"
        dfs.append(temp)

if dfs:
    df = pd.concat(dfs, ignore_index=True)
else:
    st.error("Erro crítico: nenhuma base de dados foi encontrada no servidor.")
    st.stop()

with st.sidebar:
    st.header("Navegação do Sistema")
    menu = st.selectbox("Selecione o Dashboard", [
        "📊 Painel Executivo",
        "📈 Tendências e Projeções",
        "📋 Qualidade, Evidências e Encaminhamentos",
        "⏱️ Operação e Equipe",
        "📂 Dashboard Temático & Termos",
        "🗺️ Inteligência Territorial",
        "⚠️ Risco e Reincidência",
        "👣 Jornada dos Casos",
        "📌 Análises Cruzadas para Coordenação",
        "📆 Séries Históricas Anuais",
        "🎯 Metas e Relatório Executivo",
        "ℹ️ Metodologia e Dicionário de Dados"
    ])  
    st.markdown("---")
    st.subheader("Filtros Globais")

    data_min = df["data"].min().date()
    data_max = df["data"].max().date()

    periodo = st.date_input(
        "Período de data",
        value=(data_min, data_max),
        min_value=data_min,
        max_value=data_max
    )

    opcoes = preparar_opcoes_filtros(df)

    anos = st.multiselect(
        "Anos",
        opcoes["anos"],
        default=opcoes["anos"]
    )

    tipos = st.multiselect(
        "Tipo de atendimento",
        opcoes["tipos"],
        default=opcoes["tipos"]
    )

    assuntos = st.multiselect(
        "Assuntos",
        opcoes["assuntos"]
    )

    generos = st.multiselect(
        "Gênero",
        ["Feminino", "Masculino", "Misto", "Não informado"]
    )

    comunidades = st.multiselect(
        "Comunidade",
        opcoes["comunidades"]
    )
    faixas = st.multiselect("Faixa etária", ["crianças", "adolescentes", "jovens", "adultos", "idosos"])
    caso_sensivel = st.multiselect("Caso sensível", ["Sim", "Não"])

f = df.copy()

if isinstance(periodo, tuple) and len(periodo) == 2:
    data_inicio, data_fim = periodo
    f = f[
        (f["data"].dt.date >= data_inicio) &
        (f["data"].dt.date <= data_fim)
    ]

if anos:
    f = f[f["ano"].isin(anos)]

if tipos:
    f = f[f["tipo"].isin(tipos)]

if assuntos:
    f = f[f["tema_principal"].isin(assuntos)]

if generos:
    f = f[f["genero_predominante"].isin(generos)]

if comunidades:
    f = f[f["comunidade_analise"].isin(comunidades)]

if faixas:
    colunas_faixa = {
        "crianças": "criancas",
        "adolescentes": "adolescentes",
        "jovens": "jovens",
        "adultos": "adultos",
        "idosos": "idosos"
    }

    cols_filtradas = [colunas_faixa[x] for x in faixas if colunas_faixa[x] in f.columns]

    if cols_filtradas:
        f = f[f[cols_filtradas].sum(axis=1) > 0]

if caso_sensivel:
    mapa_caso = {"Sim": True, "Não": False}
    f = f[f["caso_sensivel_flag"].isin([mapa_caso[x] for x in caso_sensivel])]

if f.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

st.sidebar.markdown("---")
st.sidebar.subheader("Exportação")

csv_filtrado = f.to_csv(index=False).encode("utf-8-sig")

st.sidebar.download_button(
    label="Baixar base filtrada",
    data=csv_filtrado,
    file_name="base_filtrada_assessoramento.csv",
    mime="text/csv"
)
# ==================================================================================
# 5. CONSTRUTOR DE PÁGINAS (DASHBOARDS)
# ==================================================================================

# --- 5.1. PAINEL EXECUTIVO ---
if menu == "📊 Painel Executivo":
    st.header("Painel Executivo")

    st.caption(
        "Visão consolidada dos principais indicadores, alertas, perfil do público e evolução dos atendimentos."
    )

    aba_resumo, aba_alertas, aba_rankings, aba_perfil, aba_evolucao = st.tabs([
        "Resumo",
        "Alertas",
        "Rankings",
        "Perfil",
        "Evolução"
    ])

    f_ordenado = f.sort_values("data")

    if len(f_ordenado) > 0:
        ultimo_mes_ano = f_ordenado["ano_mes"].iloc[-1]
        vol_mes_atual = f_ordenado[f_ordenado["ano_mes"] == ultimo_mes_ano].shape[0]

        periodos_anteriores = f_ordenado[
            f_ordenado["ano_mes"] != ultimo_mes_ano
        ]

        if not periodos_anteriores.empty:
            vol_mes_anterior = (
                periodos_anteriores
                .groupby("ano_mes")
                .size()
                .iloc[-1]
            )

            variacao_mom = round(
                (vol_mes_atual - vol_mes_anterior)
                / max(1, vol_mes_anterior) * 100,
                1
            )

            delta_mom_txt = f"{variacao_mom}% vs mês anterior"
        else:
            delta_mom_txt = "Primeiro período"
    else:
        vol_mes_atual = 0
        delta_mom_txt = "Sem dados"

    total_fem = int(f["feminino"].sum()) if "feminino" in f.columns else 0
    total_masc = int(f["masculino"].sum()) if "masculino" in f.columns else 0
    total_participantes = int(f["total_participantes"].sum()) if "total_participantes" in f.columns else 0

    with aba_resumo:
        st.subheader("📌Indicadores estratégicos")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Atendimentos",
            f.shape[0]
        )

        c2.metric(
            "Participantes",
            total_participantes
        )

        c3.metric(
            "Comunidades",
            f["comunidade_analise"].nunique()
        )

        c4.metric(
            "Encaminhamentos",
            int(f["tem_encaminhamento"].sum())
        )

        st.markdown("---")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Casos sensíveis",
            int(f["caso_sensivel_flag"].sum())
        )

        c2.metric(
            "Qualidade média",
            f"{f['ind_qualidade'].mean():.1f}%"
        )

        c3.metric(
            "Risco médio",
            f"{f['score_risco_territorial'].mean():.1f}"
        )

        c4.metric(
            "Horas técnicas",
            f"{f['duracao_horas'].sum():.1f}h"
        )

        interpretar(
            "esta visão reúne os principais indicadores de volume, qualidade, risco e carga técnica do período filtrado."
        )

    with aba_alertas:
        st.subheader("⚠️ Alertas principais")

        temas_horas = (
            f.groupby("tema_principal")["duracao_horas"]
            .sum()
            .sort_values(ascending=False)
        )

        total_horas = temas_horas.sum()

        perc_top4 = round(
            (temas_horas.head(4).sum() / total_horas * 100),
            0
        ) if total_horas > 0 else 0

        st.metric(
            "Concentração de esforço técnico",
            f"{int(perc_top4)}% das horas"
        )

        st.info(
            f"Os 4 temas que mais consomem tempo representam "
            f"{int(perc_top4)}% das horas técnicas registradas."
        )

        top_comunidade_sensivel = (
            f.groupby("comunidade_analise")["caso_sensivel_flag"]
            .sum()
            .sort_values(ascending=False)
        )

        total_sensiveis = f["caso_sensivel_flag"].sum()

        if total_sensiveis > 0 and not top_comunidade_sensivel.empty:
            perc_com_sensivel = round(
                (
                    top_comunidade_sensivel.iloc[0]
                    / total_sensiveis
                ) * 100,
                0
            )

            st.warning(
                f"A comunidade {top_comunidade_sensivel.index[0]} concentra "
                f"{int(perc_com_sensivel)}% dos casos sensíveis."
            )

        pendencias_totais = f[
            (f["doc_flag"] == False)
            | (f["drive_flag"] == False)
            | (f["tem_relatorio"] == False)
        ].shape[0]

        percentual_pendente = (
            round(
                (pendencias_totais / f.shape[0]) * 100,
                1
            )
            if f.shape[0] > 0
            else 0
        )

        if percentual_pendente > 30:
            st.error(
                f"{percentual_pendente}% dos registros possuem alguma pendência documental."
            )
        else:
            st.success(
                "A qualidade documental está em nível aceitável para os filtros atuais."
            )

    with aba_rankings:
        st.subheader("🏆 Rankings principais")

        top_temas = (
            f.groupby("tema_principal")
            .size()
            .reset_index(name="Atendimentos")
            .rename(columns={"tema_principal": "Tema"})
        )

        grafico_ranking_horizontal(
            top_temas,
            x="Atendimentos",
            y="Tema",
            titulo="Temas com maior volume de atendimentos",
            texto="os temas no topo concentram a maior parte da demanda registrada."
        )

        st.markdown("---")

        top_comunidades = (
            f.groupby("comunidade_analise")
            .size()
            .reset_index(name="Atendimentos")
            .rename(columns={"comunidade_analise": "Comunidade"})
        )

        grafico_ranking_horizontal(
            top_comunidades,
            x="Atendimentos",
            y="Comunidade",
            titulo="Comunidades com maior volume de atendimentos",
            texto="as comunidades no topo demandaram mais presença ou acompanhamento técnico."
        )

    with aba_perfil:
        st.subheader("👥 Perfil do público")

        df_genero = pd.DataFrame({
            "Sexo": ["Feminino", "Masculino"],
            "Participantes": [total_fem, total_masc]
        })

        grafico_ranking_horizontal(
            df_genero,
            x="Participantes",
            y="Sexo",
            titulo="Participantes por sexo",
            texto="a leitura considera a soma de participantes informados nos registros."
        )

        st.markdown("---")

        perfil_faixa = pd.DataFrame({
            "Faixa etária": ["Crianças", "Adolescentes", "Jovens", "Adultos", "Idosos"],
            "Participantes": [
                f["criancas"].sum() if "criancas" in f.columns else 0,
                f["adolescentes"].sum() if "adolescentes" in f.columns else 0,
                f["jovens"].sum() if "jovens" in f.columns else 0,
                f["adultos"].sum() if "adultos" in f.columns else 0,
                f["idosos"].sum() if "idosos" in f.columns else 0
            ]
        })

        perfil_faixa = perfil_faixa[
            perfil_faixa["Participantes"] > 0
        ]

        if not perfil_faixa.empty:
            grafico_ranking_horizontal(
                perfil_faixa,
                x="Participantes",
                y="Faixa etária",
                titulo="Participantes por faixa etária",
                texto="as faixas com maior volume indicam o público mais alcançado."
            )

    with aba_evolucao:
        st.subheader("📈 Evolução da demanda")

        crescimento = (
            f.groupby("ano_mes")
            .size()
            .reset_index(name="Atendimentos")
            .sort_values("ano_mes")
        )

        st.plotly_chart(
            px.line(
                crescimento.tail(12),
                x="ano_mes",
                y="Atendimentos",
                markers=True,
                title="Evolução mensal de atendimentos"
            ),
            use_container_width=True
        )

        interpretar(
            "picos indicam aumento de demanda ou maior volume de registros no período."
        )

# --- 5.3. QUALIDADE, EVIDÊNCIAS E ENCAMINHAMENTOS ---
elif menu == "📋 Qualidade, Evidências e Encaminhamentos":
    st.header("Qualidade, Evidências e Encaminhamentos")

    st.caption(
        "Acompanha qualidade documental, registros com evidência, encaminhamentos realizados "
        "e saúde geral dos registros."
    )

    aba_resumo, aba_evidencias, aba_encaminhamentos, aba_saude = st.tabs([
        "Resumo",
        "Evidências",
        "Encaminhamentos",
        "Saúde dos Registros"
    ])

    total = f.shape[0]

    perc_doc = round(f["doc_flag"].mean() * 100, 1) if total > 0 else 0
    perc_drive = round(f["drive_flag"].mean() * 100, 1) if total > 0 else 0
    perc_relatorio = round(f["tem_relatorio"].mean() * 100, 1) if total > 0 else 0
    perc_enc = round(f["tem_encaminhamento"].mean() * 100, 1) if total > 0 else 0
    qualidade_media = round(f["ind_qualidade"].mean(), 1) if total > 0 else 0

    with aba_resumo:
        st.subheader("📌 Visão geral da qualidade documental")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Com documento",
            f"{perc_doc}%",
            help="Percentual de registros com documento associado."
        )

        c2.metric(
            "No drive",
            f"{perc_drive}%",
            help="Percentual de registros salvos no drive."
        )

        c3.metric(
            "Com relatório",
            f"{perc_relatorio}%",
            help="Percentual de registros com relatório gerado."
        )

        c4.metric(
            "Com encaminhamento",
            f"{perc_enc}%",
            help="Percentual de registros que geraram encaminhamento."
        )

        interpretar(
            "percentuais altos indicam melhor qualidade de registro e maior capacidade de comprovação institucional."
        )

        st.markdown("---")

        pendencias = pd.DataFrame({
            "Indicador": ["Sem documento", "Sem drive", "Sem relatório"],
            "Registros": [
                int((f["doc_flag"] == False).sum()),
                int((f["drive_flag"] == False).sum()),
                int((f["tem_relatorio"] == False).sum())
            ]
        })

        grafico_ranking_horizontal(
            pendencias,
            x="Registros",
            y="Indicador",
            titulo="Principais pendências de evidência",
            texto="as barras maiores indicam onde a equipe deve priorizar correção ou complementação documental."
        )

    with aba_evidencias:
        st.subheader("📁 Evidências documentais")

        evidencias = pd.DataFrame({
            "Evidência": ["Documento", "Drive", "Relatório"],
            "Com evidência": [
                int(f["doc_flag"].sum()),
                int(f["drive_flag"].sum()),
                int(f["tem_relatorio"].sum())
            ],
            "Sem evidência": [
                int((f["doc_flag"] == False).sum()),
                int((f["drive_flag"] == False).sum()),
                int((f["tem_relatorio"] == False).sum())
            ]
        })

        st.dataframe(
            evidencias,
            use_container_width=True,
            hide_index=True
        )

        interpretar(
            "esta tabela mostra o volume absoluto de registros completos e incompletos por tipo de evidência."
        )

    with aba_encaminhamentos:
        st.subheader("📨 Encaminhamentos realizados")

        enc_vct = (
            f["foi_encaminhado"]
            .value_counts()
            .reset_index()
        )

        enc_vct.columns = ["Status", "Atendimentos"]

        grafico_ranking_horizontal(
            enc_vct,
            x="Atendimentos",
            y="Status",
            titulo="Status dos encaminhamentos",
            texto="mostra quantos registros geraram ou não encaminhamento."
        )

        st.markdown("---")

        demandas_encaminhadas = f[f["foi_encaminhado"] == "Encaminhado"]

        if not demandas_encaminhadas.empty:
            temas_encaminhados = (
                demandas_encaminhadas
                .groupby("tema_principal")
                .size()
                .reset_index(name="Atendimentos")
                .rename(columns={"tema_principal": "Tema"})
            )

            grafico_ranking_horizontal(
                temas_encaminhados,
                x="Atendimentos",
                y="Tema",
                titulo="Temas que mais geram encaminhamentos",
                texto="esses temas costumam exigir articulação com outras áreas, serviços ou instituições.",
                top_n=10
            )
        else:
            st.info("Não há encaminhamentos nos filtros atuais.")

    with aba_saude:
        st.subheader("🩺 Saúde dos Registros")

        st.caption(
            "Avalia a qualidade documental dos atendimentos e ajuda a identificar onde os registros precisam de reforço."
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Qualidade média",
            f"{qualidade_media}%",
            help="Média do índice de qualidade dos registros filtrados."
        )

        c2.metric(
            "Com Drive",
            f"{perc_drive}%",
            help="Percentual de registros salvos no drive."
        )

        c3.metric(
            "Com relatório",
            f"{perc_relatorio}%",
            help="Percentual de registros com relatório."
        )

        c4.metric(
            "Com documento",
            f"{perc_doc}%",
            help="Percentual de registros com documento associado."
        )

        interpretar(
            "a saúde dos registros mostra se os atendimentos estão bem documentados em drive, relatório e documentos anexados."
        )

        st.markdown("---")

        col1, col2 = st.columns(2)

        with col1:
            qualidade_assessor = (
                f.groupby("assessor1")
                .agg(
                    Qualidade=("ind_qualidade", "mean"),
                    Atendimentos=("tipo", "size")
                )
                .reset_index()
                .sort_values("Qualidade", ascending=False)
            )

            grafico_ranking_horizontal(
                qualidade_assessor,
                x="Qualidade",
                y="assessor1",
                titulo="Qualidade média por assessor",
                texto="ajuda a identificar padrões de documentação entre responsáveis.",
                top_n=12
            )

        with col2:
            qualidade_tipo = (
                f.groupby("tipo")
                .agg(
                    Qualidade=("ind_qualidade", "mean"),
                    Atendimentos=("tipo", "size")
                )
                .reset_index()
                .sort_values("Qualidade", ascending=False)
            )

            grafico_ranking_horizontal(
                qualidade_tipo,
                x="Qualidade",
                y="tipo",
                titulo="Qualidade média por tipo de agenda",
                texto="mostra quais tipos de atendimento tendem a gerar registros mais completos.",
                top_n=12
            )

        st.markdown("---")

        st.subheader("📋 Tabela de saúde documental")

        tabela_saude = (
            f.groupby("assessor1")
            .agg(
                Atendimentos=("tipo", "size"),
                Qualidade=("ind_qualidade", "mean"),
                Drive=("drive_flag", "mean"),
                Relatório=("tem_relatorio", "mean"),
                Documento=("doc_flag", "mean")
            )
            .reset_index()
        )

        tabela_saude["Drive"] *= 100
        tabela_saude["Relatório"] *= 100
        tabela_saude["Documento"] *= 100

        st.dataframe(
            tabela_saude.rename(columns={
                "assessor1": "Assessor",
                "Qualidade": "Qualidade média",
                "Drive": "% Drive",
                "Relatório": "% Relatório",
                "Documento": "% Documento"
            }).style.format({
                "Qualidade média": "{:.1f}%",
                "% Drive": "{:.1f}%",
                "% Relatório": "{:.1f}%",
                "% Documento": "{:.1f}%"
            }),
            use_container_width=True,
            hide_index=True
        )

# --- 5.4. OPERAÇÃO E EQUIPE ---
elif menu == "⏱️ Operação e Equipe":
    st.header("⏱️ Operação e Equipe")

    st.caption(
        "Mostra sazonalidade, formatos de atendimento, horas técnicas, distribuição da carga e atuação em parceria."
    )

    aba_resumo, aba_sazonalidade, aba_logistica, aba_carga, aba_parcerias = st.tabs([
        "📌 Resumo",
        "📅 Sazonalidade",
        "🚗 Logística",
        "👥 Carga técnica",
        "🤝 Parcerias"
    ])

    equipe = (
        f.groupby("assessor1")
        .agg(
            Atendimentos=("tipo", "size"),
            Horas_tecnicas=("duracao_horas", "sum"),
            Sobrecarga=("ind_sobrecarga", "sum"),
            Complexidade_media=("ind_complexidade", "mean")
        )
        .reset_index()
        .rename(columns={
            "assessor1": "Assessor",
            "Horas_tecnicas": "Horas técnicas",
            "Complexidade_media": "Complexidade média"
        })
        .sort_values("Sobrecarga", ascending=False)
    )

    with aba_resumo:
        st.subheader("📌 Visão geral operacional")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Atendimentos",
            f.shape[0],
            help="Total de atendimentos registrados no período filtrado."
        )

        c2.metric(
            "Horas técnicas",
            f"{f['duracao_horas'].sum():.1f}h",
            help="Soma de horas técnicas registradas."
        )

        c3.metric(
            "Tempo médio",
            f"{f['duracao_horas'].mean():.1f}h",
            help="Tempo médio por atendimento."
        )

        c4.metric(
            "Assessores ativos",
            equipe["Assessor"].nunique(),
            help="Quantidade de assessores com atendimentos no período."
        )

        interpretar(
            "esta visão mostra o esforço operacional necessário para sustentar os atendimentos e como a carga está distribuída entre a equipe."
        )

        st.markdown("---")

        grafico_ranking_horizontal(
            equipe,
            x="Atendimentos",
            y="Assessor",
            titulo="Atendimentos por assessor",
            texto="barras maiores indicam maior volume de registros vinculados ao assessor.",
            top_n=15
        )

    with aba_sazonalidade:
        st.subheader("📅 Sazonalidade dos atendimentos")

        sazonalidade_mes = (
            f.groupby("mes")
            .size()
            .reset_index(name="Atendimentos")
        )

        st.plotly_chart(
            px.line(
                sazonalidade_mes,
                x="mes",
                y="Atendimentos",
                markers=True,
                title="Volume de atendimentos por mês"
            ),
            use_container_width=True
        )

        interpretar(
            "meses com pico podem indicar maior pressão sobre equipe, agenda ou demandas territoriais."
        )

        st.markdown("---")

        ordem_dias = [
            "Segunda-feira", "Terça-feira", "Quarta-feira",
            "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"
        ]

        dias_criticos = (
            f.groupby("dia_semana")
            .size()
            .reindex(ordem_dias, fill_value=0)
            .reset_index(name="Atendimentos")
            .rename(columns={"dia_semana": "Dia da semana"})
        )

        grafico_ranking_horizontal(
            dias_criticos,
            x="Atendimentos",
            y="Dia da semana",
            titulo="Dias com maior volume de atendimento",
            texto="os dias no topo indicam maior pressão de agenda.",
            top_n=7
        )

    with aba_logistica:
        st.subheader("🚗 Logística dos atendimentos")

        matriz_log = (
            f.groupby("formato")
            .agg(
                Atendimentos=("tipo", "size"),
                Complexidade_media=("ind_complexidade", "mean"),
                Horas_tecnicas=("duracao_horas", "sum")
            )
            .reset_index()
            .rename(columns={
                "formato": "Formato",
                "Complexidade_media": "Complexidade média",
                "Horas_tecnicas": "Horas técnicas"
            })
        )

        st.dataframe(
            matriz_log.style.format({
                "Complexidade média": "{:.1f}",
                "Horas técnicas": "{:.1f}h"
            }),
            use_container_width=True,
            hide_index=True
        )

        interpretar(
            "formatos com maior complexidade média ou mais horas exigem maior planejamento operacional."
        )

    with aba_carga:
        st.subheader("👥 Carga técnica da equipe")

        grafico_ranking_horizontal(
            equipe,
            x="Sobrecarga",
            y="Assessor",
            titulo="Ranking de sobrecarga técnica",
            texto="a sobrecarga combina volume, duração, complexidade e peso adicional de casos sensíveis.",
            top_n=15
        )

        st.markdown("---")

        grafico_ranking_horizontal(
            equipe,
            x="Horas técnicas",
            y="Assessor",
            titulo="Ranking de horas técnicas",
            texto="mostra quem acumulou mais tempo registrado em atendimentos no período filtrado.",
            top_n=15
        )

        st.markdown("---")

        grafico_ranking_horizontal(
            equipe,
            x="Complexidade média",
            y="Assessor",
            titulo="Ranking de complexidade média",
            texto="assessores no topo aparecem associados a atendimentos mais complexos em média.",
            top_n=15
        )

        st.markdown("---")

        st.subheader("📋 Tabela de leitura da carga")

        equipe_tabela = equipe.sort_values(
            ["Sobrecarga", "Horas técnicas", "Complexidade média"],
            ascending=False
        ).copy()

        st.dataframe(
            equipe_tabela.style.format({
                "Horas técnicas": "{:.1f}h",
                "Sobrecarga": "{:.1f}",
                "Complexidade média": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )

        interpretar(
            "use a tabela para comparar volume, tempo, complexidade e sobrecarga de forma conjunta."
        )

    with aba_parcerias:
        st.subheader("🤝 Atuação em parceria")

        if "assessor2" in f.columns:
            duplas_validas = f.dropna(subset=["assessor1", "assessor2"]).copy()
            duplas_validas["assessor1"] = duplas_validas["assessor1"].astype(str).str.strip()
            duplas_validas["assessor2"] = duplas_validas["assessor2"].astype(str).str.strip()

            duplas_validas = duplas_validas[
                (duplas_validas["assessor1"] != "")
                & (duplas_validas["assessor2"] != "")
                & (duplas_validas["assessor1"].str.lower() != "nan")
                & (duplas_validas["assessor2"].str.lower() != "nan")
            ]

            if not duplas_validas.empty:
                duplas_validas["Parceria"] = duplas_validas.apply(
                    lambda row: " + ".join(
                        sorted([row["assessor1"], row["assessor2"]])
                    ),
                    axis=1
                )

                ranking_duplas = (
                    duplas_validas
                    .groupby("Parceria")
                    .size()
                    .reset_index(name="Atendimentos")
                    .sort_values("Atendimentos", ascending=False)
                )

                grafico_ranking_horizontal(
                    ranking_duplas,
                    x="Atendimentos",
                    y="Parceria",
                    titulo="Parcerias mais frequentes",
                    texto="mostra quais duplas aparecem mais vezes nos atendimentos.",
                    top_n=10
                )
            else:
                st.info("Não há registros de atuação em dupla nos filtros atuais.")
        else:
            st.info("A base não possui coluna de segundo assessor.")

# --- 5.6. DASHBOARD TEMÁTICO & TERMOS ---
elif menu == "📂 Dashboard Temático & Termos":
    st.header("Temas e Termos")
    st.caption("Mostra os temas que mais demandam tempo técnico e os termos mais frequentes nos textos.")

    aba_resumo, aba_temas, aba_termos = st.tabs([
        "Resumo",
        "Temas",
        "Termos-chave"
    ])

    temas = (
        f.groupby("tema_principal")
        .agg(
            atendimentos=("tipo", "size"),
            horas_totais=("duracao_horas", "sum"),
            participantes_totais=("total_participantes", "sum"),
            eficiencia_media=("ind_eficiencia", "mean")
        )
        .reset_index()
        .rename(columns={
            "tema_principal": "Tema",
            "atendimentos": "Atendimentos",
            "horas_totais": "Horas técnicas",
            "participantes_totais": "Participantes",
            "eficiencia_media": "Participantes por hora"
        })
        .sort_values("Horas técnicas", ascending=False)
    )

    with aba_resumo:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Temas registrados", temas["Tema"].nunique())
        c2.metric("Tema mais frequente", temas.sort_values("Atendimentos", ascending=False)["Tema"].iloc[0] if not temas.empty else "Sem dados")
        c3.metric("Horas técnicas", f"{temas['Horas técnicas'].sum():.1f}h")
        c4.metric("Participantes", int(temas["Participantes"].sum()))

        interpretar(
            "o resumo mostra a concentração temática do trabalho no período filtrado."
        )

    with aba_temas:
        grafico_ranking_horizontal(
            temas,
            x="Horas técnicas",
            y="Tema",
            titulo="Temas que mais consomem tempo técnico",
            texto="quanto maior a barra, maior o esforço técnico dedicado ao tema.",
            top_n=12
        )

        st.markdown("---")

        st.dataframe(
            temas.style.format({
                "Horas técnicas": "{:.1f}h",
                "Participantes": "{:.0f}",
                "Participantes por hora": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )

    with aba_termos:
        col_txt1, col_txt2 = st.columns(2)

        with col_txt1:
            st.subheader("Termos em encaminhamentos")

            if "encaminhamentos" in f.columns:
                df_termos_enc = extrair_palavras_chave(f["encaminhamentos"])

                if not df_termos_enc.empty:
                    grafico_ranking_horizontal(
                        df_termos_enc,
                        x="Frequência",
                        y="Termo",
                        titulo="Palavras mais frequentes em encaminhamentos",
                        texto="termos recorrentes ajudam a identificar padrões de demanda e resposta técnica."
                    )
                else:
                    st.info("Não há termos suficientes em encaminhamentos.")
            else:
                st.info("A base não possui coluna de encaminhamentos.")

        with col_txt2:
            st.subheader("Termos em relatórios")

            if "relatorio" in f.columns:
                df_termos_rel = extrair_palavras_chave(f["relatorio"])

                if not df_termos_rel.empty:
                    grafico_ranking_horizontal(
                        df_termos_rel,
                        x="Frequência",
                        y="Termo",
                        titulo="Palavras mais frequentes em relatórios",
                        texto="termos recorrentes ajudam a interpretar os assuntos mais registrados nos textos técnicos."
                    )
                else:
                    st.info("Não há termos suficientes em relatórios.")
            else:
                st.info("A base não possui coluna de relatório.")

# --- 5.7. INTELIGÊNCIA TERRITORIAL ---
elif menu == "🗺️ Inteligência Territorial":
    st.header("Inteligência Territorial")

    st.caption(
        "Reúne volume, risco, cobertura recente, lacunas de acompanhamento e prioridades por comunidade."
    )

    aba_resumo, aba_mapa, aba_cobertura, aba_prioridades, aba_tabela = st.tabs([
        "Resumo",
        "Mapa",
        "Cobertura",
        "Prioridades",
        "Tabela"
    ])

    comunidade_col = "comunidade_analise" if "comunidade_analise" in f.columns else "local"

    territorios = (
        f.groupby(comunidade_col)
        .agg(
            Atendimentos=("tipo", "size"),
            Participantes=("total_participantes", "sum"),
            ultima_data=("data", "max"),
            Casos_sensiveis=("caso_sensivel_flag", "sum"),
            Risco_medio=("score_risco_territorial", "mean")
        )
        .reset_index()
        .rename(columns={comunidade_col: "Comunidade"})
    )

    territorios = territorios[
        territorios["Comunidade"].astype(str).str.lower() != "todas as comunidades"
    ]

    data_referencia = f["data"].max()

    territorios["Dias sem atendimento"] = (
        data_referencia - territorios["ultima_data"]
    ).dt.days

    territorios["Status"] = np.select(
        [
            territorios["Dias sem atendimento"] <= 30,
            territorios["Dias sem atendimento"].between(31, 60),
            territorios["Dias sem atendimento"] > 60
        ],
        [
            "Cobertura recente",
            "Atenção",
            "Sem atendimento recente"
        ],
        default="Não classificado"
    )

    territorios["Índice de prioridade"] = (
        territorios["Dias sem atendimento"].fillna(0) * 0.35
        + territorios["Risco_medio"].fillna(0) * 0.40
        + territorios["Casos_sensiveis"].fillna(0) * 0.25
    )

    with aba_resumo:
        st.subheader("📌 Visão geral territorial")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Comunidades",
            territorios["Comunidade"].nunique(),
            help="Quantidade de comunidades com registros na base filtrada."
        )

        c2.metric(
            "Atendimentos",
            int(territorios["Atendimentos"].sum()),
            help="Total de atendimentos registrados nas comunidades filtradas."
        )

        c3.metric(
            "Participantes",
            int(territorios["Participantes"].sum()),
            help="Soma de participantes registrados nos atendimentos."
        )

        c4.metric(
            "Sem atendimento recente",
            int((territorios["Status"] == "Sem atendimento recente").sum()),
            help="Comunidades sem atendimento há mais de 60 dias."
        )

        interpretar(
            "a inteligência territorial mostra onde o trabalho está mais concentrado, onde há maior risco e quais comunidades precisam de atenção."
        )

        st.markdown("---")

        ranking_volume = territorios.sort_values(
            "Atendimentos",
            ascending=False
        )

        grafico_ranking_horizontal(
            ranking_volume,
            x="Atendimentos",
            y="Comunidade",
            titulo="Comunidades com maior volume de atendimentos",
            texto="as comunidades no topo tiveram maior presença ou demanda registrada.",
            top_n=12
        )

    with aba_mapa:
        st.subheader("🗺️ Mapa territorial")

        st.info(
            "O mapa usa coordenadas cadastradas no arquivo coordenadas_comunidades.xlsx "
            "Comunidades sem coordenada aparecem no aviso e ficam fora do mapa."
        )

        try:
            coords = pd.read_excel("coordenadas_comunidades.xlsx")

            mapa = territorios.merge(
                coords,
                on="Comunidade",
                how="left"
            )

            sem_coordenada = mapa[
                mapa["Latitude"].isna() | mapa["Longitude"].isna()
            ]

            if not sem_coordenada.empty:
                st.warning(
                    f"{sem_coordenada.shape[0]} comunidade(s) ainda não possuem coordenadas cadastradas."
                )

            mapa = mapa.dropna(subset=["Latitude", "Longitude"])

            if not mapa.empty:
                fig_mapa = px.scatter_mapbox(
                    mapa,
                    lat="Latitude",
                    lon="Longitude",
                    size="Atendimentos",
                    color="Risco_medio",
                    hover_name="Comunidade",
                    hover_data={
                        "Atendimentos": True,
                        "Participantes": True,
                        "Casos_sensiveis": True,
                        "Risco_medio": ":.1f",
                        "Dias sem atendimento": True,
                        "Status": True,
                        "Latitude": False,
                        "Longitude": False
                    },
                    size_max=35,
                    zoom=9,
                    title="Mapa territorial"
                )

                fig_mapa.update_layout(
                    mapbox_style="open-street-map",
                    margin=dict(l=0, r=0, t=40, b=0)
                )

                st.plotly_chart(fig_mapa, use_container_width=True)

                interpretar(
                    "círculos maiores indicam mais atendimentos. A cor representa o risco territorial médio."
                )

            else:
                st.warning(
                    "Nenhuma comunidade dos filtros atuais possui coordenadas válidas."
                )

        except FileNotFoundError:
            st.error(
                "Arquivo coordenadas_comunidades.csv não encontrado. "
                "Crie esse arquivo na mesma pasta do app.py para ativar o mapa real."
            )

    with aba_cobertura:
        st.subheader("🧭 Cobertura territorial")

        status_count = (
            territorios["Status"]
            .value_counts()
            .reset_index()
        )

        status_count.columns = ["Status", "Comunidades"]

        grafico_ranking_horizontal(
            status_count,
            x="Comunidades",
            y="Status",
            titulo="Status da cobertura territorial",
            texto="mostra quantas comunidades estão com acompanhamento recente, em atenção ou sem atendimento recente."
        )

        st.markdown("---")

        sem_atendimento = territorios[
            territorios["Status"] == "Sem atendimento recente"
        ].sort_values(
            "Dias sem atendimento",
            ascending=False
        )

        if not sem_atendimento.empty:
            grafico_ranking_horizontal(
                sem_atendimento,
                x="Dias sem atendimento",
                y="Comunidade",
                titulo="Comunidades há mais tempo sem atendimento",
                texto="quanto maior a barra, mais tempo a comunidade está sem registro de atendimento.",
                top_n=12
            )
        else:
            st.success(
                "Todas as comunidades filtradas possuem atendimento recente ou estão dentro da faixa de atenção."
            )

    with aba_prioridades:
        st.subheader("🚨 Prioridades territoriais")

        prioridades = territorios.sort_values(
            ["Índice de prioridade", "Dias sem atendimento", "Risco_medio"],
            ascending=False
        )

        grafico_ranking_horizontal(
            prioridades,
            x="Índice de prioridade",
            y="Comunidade",
            titulo="Ranking de prioridade territorial",
            texto="o índice combina dias sem atendimento, risco médio e casos sensíveis para orientar priorização.",
            top_n=12
        )

        st.markdown("---")

        risco = territorios.sort_values(
            "Risco_medio",
            ascending=False
        )

        grafico_ranking_horizontal(
            risco,
            x="Risco_medio",
            y="Comunidade",
            titulo="Comunidades com maior risco médio",
            texto="risco médio alto indica maior atenção técnica no território.",
            top_n=12
        )

    with aba_tabela:
        st.subheader("📋 Tabela territorial consolidada")

        st.caption(
            "Use esta tabela para verificar volume, último atendimento, risco, casos sensíveis e prioridade territorial."
        )

        tabela_territorial = territorios.sort_values(
            ["Índice de prioridade", "Dias sem atendimento", "Risco_medio"],
            ascending=False
        ).copy()

        tabela_territorial["Último atendimento"] = tabela_territorial["ultima_data"].dt.strftime("%d/%m/%Y")

        tabela_territorial = tabela_territorial.drop(columns=["ultima_data"])

        st.dataframe(
            tabela_territorial.rename(columns={
                "Casos_sensiveis": "Casos sensíveis",
                "Risco_medio": "Risco médio"
            }).style.format({
                "Participantes": "{:.0f}",
                "Risco médio": "{:.1f}",
                "Índice de prioridade": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )

# --- 5.8. RISCO E REINCIDÊNCIA ---
elif menu == "⚠️ Risco e Reincidência":
    st.header("⚠️ Risco e Reincidência")

    st.caption(
        "Mostra casos sensíveis, risco territorial, temas críticos e padrões de repetição das demandas."
    )

    aba_resumo, aba_casos, aba_risco, aba_reincidencia, aba_tabela = st.tabs([
        "Resumo",
        "Casos sensíveis",
        "Risco territorial",
        "Reincidência",
        "Tabela"
    ])

    sensiv = f[f["caso_sensivel_flag"] == True].copy()

    ranking_risco = (
        f.groupby("comunidade_analise")
        .agg(
            Risco_medio=("score_risco_territorial", "mean"),
            Atendimentos=("tipo", "size"),
            Casos_sensiveis=("caso_sensivel_flag", "sum"),
            Reincidencia_media=("ind_reincidencia", "mean")
        )
        .reset_index()
        .rename(columns={
            "comunidade_analise": "Comunidade",
            "Risco_medio": "Risco médio",
            "Casos_sensiveis": "Casos sensíveis",
            "Reincidencia_media": "Reincidência média"
        })
        .sort_values("Risco médio", ascending=False)
    )

    ranking_risco = ranking_risco[
        ranking_risco["Comunidade"].astype(str).str.lower() != "todas as comunidades"
    ]

    reincidencia_tema = (
        f.groupby("tema_principal")
        .agg(
            Atendimentos=("tipo", "size"),
            Reincidencia_media=("ind_reincidencia", "mean"),
            Casos_sensiveis=("caso_sensivel_flag", "sum")
        )
        .reset_index()
        .rename(columns={
            "tema_principal": "Tema",
            "Reincidencia_media": "Reincidência média",
            "Casos_sensiveis": "Casos sensíveis"
        })
        .sort_values(
            ["Reincidência média", "Atendimentos"],
            ascending=False
        )
    )

    with aba_resumo:
        st.subheader("📌 Visão geral de risco")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Casos sensíveis",
            int(f["caso_sensivel_flag"].sum()),
            help="Total de registros classificados como caso sensível."
        )

        c2.metric(
            "Risco médio",
            f"{f['score_risco_territorial'].mean():.1f}",
            help="Média do score de risco territorial na base filtrada."
        )

        c3.metric(
            "Comunidades com risco",
            ranking_risco[ranking_risco["Risco médio"] > 0]["Comunidade"].nunique(),
            help="Quantidade de comunidades com algum nível de risco identificado."
        )

        c4.metric(
            "Temas sensíveis",
            sensiv["tema_principal"].nunique() if not sensiv.empty else 0,
            help="Quantidade de temas diferentes presentes em casos sensíveis."
        )

        interpretar(
            "casos sensíveis não indicam erro. Eles sinalizam registros que exigem maior atenção, acompanhamento ou articulação."
        )

    with aba_casos:
        st.subheader("⚠️ Casos sensíveis por tema")

        if not sensiv.empty:
            temas_sensiveis = (
                sensiv.groupby("tema_principal")
                .size()
                .reset_index(name="Casos sensíveis")
                .rename(columns={"tema_principal": "Tema"})
            )

            grafico_ranking_horizontal(
                temas_sensiveis,
                x="Casos sensíveis",
                y="Tema",
                titulo="Temas com maior volume de casos sensíveis",
                texto="os temas no topo concentram maior atenção técnica sensível.",
                top_n=12
            )
        else:
            st.success("Não há casos sensíveis nos filtros atuais.")

    with aba_risco:
        st.subheader("🚨 Risco territorial")

        grafico_ranking_horizontal(
            ranking_risco,
            x="Risco médio",
            y="Comunidade",
            titulo="Comunidades com maior risco territorial",
            texto="o risco médio combina vulnerabilidade, reincidência e presença de casos sensíveis.",
            top_n=12
        )

        st.markdown("---")

        grafico_ranking_horizontal(
            ranking_risco.sort_values(
                ["Casos sensíveis", "Risco médio"],
                ascending=False
            ),
            x="Casos sensíveis",
            y="Comunidade",
            titulo="Comunidades com mais casos sensíveis",
            texto="ajuda a localizar onde há maior concentração de registros que exigem atenção técnica.",
            top_n=12
        )

    with aba_reincidencia:
        st.subheader("🔄 Reincidência das demandas")

        grafico_ranking_horizontal(
            reincidencia_tema,
            x="Reincidência média",
            y="Tema",
            titulo="Temas com maior reincidência média",
            texto="temas reincidentes indicam demandas que retornam com frequência e podem exigir ação estruturante.",
            top_n=12
        )

        st.markdown("---")

        reincidencia_comunidade = ranking_risco.sort_values(
            ["Reincidência média", "Atendimentos"],
            ascending=False
        )

        grafico_ranking_horizontal(
            reincidencia_comunidade,
            x="Reincidência média",
            y="Comunidade",
            titulo="Comunidades com maior reincidência média",
            texto="mostra onde as demandas tendem a se repetir com mais intensidade.",
            top_n=12
        )

        interpretar(
            "reincidência alta pode indicar que uma demanda não está sendo resolvida de forma definitiva ou que há necessidade de pactuação com outras áreas."
        )

    with aba_tabela:
        st.subheader("📋 Tabela consolidada de risco e reincidência")

        st.dataframe(
            ranking_risco.style.format({
                "Risco médio": "{:.1f}",
                "Atendimentos": "{:.0f}",
                "Casos sensíveis": "{:.0f}",
                "Reincidência média": "{:.1f}"
            }),
            use_container_width=True,
            hide_index=True
        )
        
# --- 5.9.1. JORNADA DOS CASOS ---
elif menu == "👣 Jornada dos Casos":
    st.header("Jornada dos Casos")
    st.caption("Acompanha histórico, recorrência e risco dos casos a partir dos identificadores disponíveis na base.")

    aba_resumo, aba_linha_tempo, aba_recorrencia, aba_detalhes = st.tabs([
        "Resumo",
        "Linha do tempo",
        "Recorrência",
        "Detalhamento"
    ])

    total_registros = f.shape[0]
    registros_com_id = int(f["tem_id_caso"].sum()) if "tem_id_caso" in f.columns else 0
    perc_com_id = round((registros_com_id / total_registros) * 100, 1) if total_registros > 0 else 0

    if "tem_id_caso" not in f.columns or "id_caso" not in f.columns:
        st.error(
            "As colunas id_caso e tem_id_caso ainda não foram criadas no carregamento da base. "
            "Inclua a criação dessas colunas dentro da função load_data."
        )

    else:
        base_jornada = f[f["tem_id_caso"] == True].copy()

        if base_jornada.empty:
            st.warning(
                "Não há IDs válidos nos filtros atuais. A análise de jornada precisa de ID preenchido."
            )

            with aba_resumo:
                c1, c2, c3 = st.columns(3)
                c1.metric("Registros filtrados", total_registros)
                c2.metric("Registros com ID", registros_com_id)
                c3.metric("Cobertura de ID", f"{perc_com_id}%")

                interpretar(
                    "sem ID preenchido, o sistema não consegue agrupar registros de um mesmo caso com segurança."
                )

        else:
            jornada = (
                base_jornada.groupby("id_caso")
                .agg(
                    Atendimentos=("tipo", "size"),
                    primeira_data=("data", "min"),
                    ultima_data=("data", "max"),
                    Comunidade=("comunidade_analise", lambda x: x.mode().iloc[0] if not x.mode().empty else "Não informado"),
                    Tema=("tema_principal", lambda x: x.mode().iloc[0] if not x.mode().empty else "Não informado"),
                    Casos_sensiveis=("caso_sensivel_flag", "sum"),
                    Encaminhamentos=("tem_encaminhamento", "sum"),
                    Documentos=("doc_flag", "sum"),
                    Relatorios=("tem_relatorio", "sum"),
                    Complexidade_media=("ind_complexidade", "mean"),
                    Risco_medio=("score_risco_territorial", "mean")
                )
                .reset_index()
                .rename(columns={"id_caso": "ID do caso"})
            )

            jornada["Dias em acompanhamento"] = (
                jornada["ultima_data"] - jornada["primeira_data"]
            ).dt.days

            jornada["Status"] = np.select(
                [
                    (jornada["Atendimentos"] >= 3) & (jornada["Casos_sensiveis"] > 0),
                    jornada["Atendimentos"] >= 3,
                    jornada["Atendimentos"] == 2,
                    jornada["Atendimentos"] == 1
                ],
                [
                    "Recorrente e sensível",
                    "Recorrente",
                    "Retorno pontual",
                    "Atendimento único"
                ],
                default="Não classificado"
            )

            with aba_resumo:
                c1, c2, c3, c4 = st.columns(4)

                c1.metric("Casos com ID", jornada.shape[0])
                c2.metric("Registros com ID", registros_com_id)
                c3.metric("Cobertura de ID", f"{perc_com_id}%")
                c4.metric("Casos recorrentes", int((jornada["Atendimentos"] >= 2).sum()))

                interpretar(
                    "a cobertura de ID mostra quanto da base filtrada pode ser analisada como jornada real de caso."
                )

                st.markdown("---")

                status_count = (
                    jornada["Status"]
                    .value_counts()
                    .reset_index()
                )
                status_count.columns = ["Status", "Quantidade"]

                grafico_ranking_horizontal(
                    status_count,
                    x="Quantidade",
                    y="Status",
                    titulo="Classificação das jornadas",
                    texto="os grupos com maior quantidade mostram o padrão dominante de acompanhamento."
                )

            with aba_linha_tempo:
                eventos_mes = (
                    base_jornada
                    .groupby("ano_mes")
                    .size()
                    .reset_index(name="Atendimentos")
                    .sort_values("ano_mes")
                )

                st.plotly_chart(
                    px.line(
                        eventos_mes,
                        x="ano_mes",
                        y="Atendimentos",
                        markers=True,
                        title="Evolução mensal dos registros com ID"
                    ),
                    use_container_width=True
                )

                interpretar(
                    "picos indicam aumento de registros identificados e podem representar maior acompanhamento de casos no período."
                )

                st.markdown("---")

                casos_novos_mes = (
                    jornada
                    .groupby(jornada["primeira_data"].dt.to_period("M").astype(str))
                    .size()
                    .reset_index(name="Casos novos")
                    .rename(columns={"primeira_data": "ano_mes"})
                )

                st.plotly_chart(
                    px.line(
                        casos_novos_mes,
                        x="ano_mes",
                        y="Casos novos",
                        markers=True,
                        title="Entrada de novos casos por mês"
                    ),
                    use_container_width=True
                )

                interpretar(
                    "mostra em quais meses novos IDs começaram a aparecer na base."
                )

            with aba_recorrencia:
                recorrentes = jornada.sort_values(
                    ["Atendimentos", "Casos_sensiveis", "Risco_medio"],
                    ascending=False
                ).head(15)

                grafico_ranking_horizontal(
                    recorrentes,
                    x="Atendimentos",
                    y="ID do caso",
                    titulo="Casos com maior recorrência",
                    texto="quanto maior a barra, maior a quantidade de registros vinculados ao mesmo ID."
                )

                st.markdown("---")

                temas_recorrentes = (
                    jornada[jornada["Atendimentos"] >= 2]
                    .groupby("Tema")
                    .size()
                    .reset_index(name="Casos recorrentes")
                )

                if not temas_recorrentes.empty:
                    grafico_ranking_horizontal(
                        temas_recorrentes,
                        x="Casos recorrentes",
                        y="Tema",
                        titulo="Temas mais presentes em casos recorrentes",
                        texto="temas no topo aparecem com mais frequência entre casos que retornam ao atendimento.",
                        top_n=10
                    )
                else:
                    st.info("Não há casos recorrentes nos filtros atuais.")

            with aba_detalhes:
                st.subheader("Tabela de acompanhamento por ID")
                st.caption("Use esta tabela para localizar casos recorrentes, sensíveis ou com maior tempo de acompanhamento.")

                jornada_exibir = jornada.sort_values(
                    ["Atendimentos", "Casos_sensiveis", "Risco_medio"],
                    ascending=False
                ).copy()

                jornada_exibir["Primeiro registro"] = jornada_exibir["primeira_data"].dt.strftime("%d/%m/%Y")
                jornada_exibir["Último registro"] = jornada_exibir["ultima_data"].dt.strftime("%d/%m/%Y")

                jornada_exibir = jornada_exibir.drop(columns=["primeira_data", "ultima_data"])

                st.dataframe(
                    jornada_exibir.rename(columns={
                        "Casos_sensiveis": "Casos sensíveis",
                        "Relatorios": "Relatórios",
                        "Complexidade_media": "Complexidade média",
                        "Risco_medio": "Risco médio"
                    }).style.format({
                        "Complexidade média": "{:.1f}",
                        "Risco médio": "{:.1f}"
                    }),
                    use_container_width=True,
                    hide_index=True
                )

# --- 5.10. ANÁLISES CRUZADAS PARA COORDENAÇÃO ---
elif menu == "📌 Análises Cruzadas para Coordenação":
    st.header("Análises Cruzadas para Coordenação")
    st.caption("Leituras estratégicas cruzando período, comunidade, assunto, público, casos sensíveis e origem da base.")

    aba_resumo, aba_publico, aba_comunidade, aba_sensiveis, aba_matrizes, aba_base = st.tabs([
        "Resumo",
        "Público",
        "Comunidade e tema",
        "Casos sensíveis",
        "Matrizes",
        "Base"
    ])

    comunidade_col = "comunidade_analise" if "comunidade_analise" in f.columns else "local"

    comunidade_mais_atendida = (
        f[comunidade_col].value_counts().idxmax()
        if comunidade_col in f.columns and not f[comunidade_col].dropna().empty
        else "Não informado"
    )

    assunto_mais_frequente = (
        f["tema_principal"].value_counts().idxmax()
        if "tema_principal" in f.columns and not f["tema_principal"].dropna().empty
        else "Não informado"
    )

    tipo_mais_frequente = (
        f["tipo"].value_counts().idxmax()
        if "tipo" in f.columns and not f["tipo"].dropna().empty
        else "Não informado"
    )

    total_casos_sensiveis = int(f["caso_sensivel_flag"].sum()) if "caso_sensivel_flag" in f.columns else 0
    total_participantes = int(f["total_participantes"].sum()) if "total_participantes" in f.columns else 0

    with aba_resumo:
        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Atendimentos", f.shape[0])
        c2.metric("Participantes", total_participantes)
        c3.metric("Casos sensíveis", total_casos_sensiveis)
        c4.metric("Comunidades", f[comunidade_col].nunique() if comunidade_col in f.columns else 0)

        st.info(
            f"No período filtrado, a comunidade com maior volume foi {comunidade_mais_atendida}. "
            f"O tema mais frequente foi {assunto_mais_frequente}. "
            f"O tipo de atendimento mais recorrente foi {tipo_mais_frequente}."
        )

        interpretar(
            "esta tela cruza dimensões importantes para apoiar decisão de coordenação, priorização territorial e leitura técnica."
        )

    with aba_publico:
        col1, col2 = st.columns(2)

        with col1:
            perfil_faixa = pd.DataFrame({
                "Faixa etária": ["Crianças", "Adolescentes", "Jovens", "Adultos", "Idosos"],
                "Participantes": [
                    f["criancas"].sum() if "criancas" in f.columns else 0,
                    f["adolescentes"].sum() if "adolescentes" in f.columns else 0,
                    f["jovens"].sum() if "jovens" in f.columns else 0,
                    f["adultos"].sum() if "adultos" in f.columns else 0,
                    f["idosos"].sum() if "idosos" in f.columns else 0,
                ]
            })

            perfil_faixa = perfil_faixa[perfil_faixa["Participantes"] > 0]

            if not perfil_faixa.empty:
                grafico_ranking_horizontal(
                    perfil_faixa,
                    x="Participantes",
                    y="Faixa etária",
                    titulo="Participantes por faixa etária",
                    texto="mostra quais faixas etárias concentram maior participação nos atendimentos filtrados.",
                    top_n=5
                )
            else:
                st.info("Não há dados de faixa etária para os filtros selecionados.")

        with col2:
            perfil_genero = pd.DataFrame({
                "Gênero": ["Feminino", "Masculino"],
                "Participantes": [
                    f["feminino"].sum() if "feminino" in f.columns else 0,
                    f["masculino"].sum() if "masculino" in f.columns else 0,
                ]
            })

            perfil_genero = perfil_genero[perfil_genero["Participantes"] > 0]

            if not perfil_genero.empty:
                grafico_ranking_horizontal(
                    perfil_genero,
                    x="Participantes",
                    y="Gênero",
                    titulo="Participantes por gênero",
                    texto="mostra a distribuição de participantes por gênero informado.",
                    top_n=2
                )
            else:
                st.info("Não há dados de gênero para os filtros selecionados.")

    with aba_comunidade:
        st.subheader("Assuntos mais recorrentes por comunidade")

        matriz_assuntos = (
            f.groupby([comunidade_col, "tema_principal"])
            .size()
            .reset_index(name="Atendimentos")
            .rename(columns={
                comunidade_col: "Comunidade",
                "tema_principal": "Tema"
            })
            .sort_values("Atendimentos", ascending=False)
        )

        if not matriz_assuntos.empty:
            top_matriz = matriz_assuntos.head(20)

            st.plotly_chart(
                px.bar(
                    top_matriz,
                    x="Atendimentos",
                    y="Tema",
                    color="Comunidade",
                    orientation="h",
                    title="Top assuntos por comunidade",
                    text_auto=True
                ),
                use_container_width=True
            )

            interpretar(
                "este gráfico mostra quais temas aparecem com mais força dentro de cada comunidade."
            )

            st.markdown("---")

            ranking_comunidades = (
                f.groupby(comunidade_col)
                .size()
                .reset_index(name="Atendimentos")
                .rename(columns={comunidade_col: "Comunidade"})
                .sort_values("Atendimentos", ascending=False)
            )

            grafico_ranking_horizontal(
                ranking_comunidades,
                x="Atendimentos",
                y="Comunidade",
                titulo="Comunidades com maior volume de atendimento",
                texto="as comunidades no topo tiveram maior quantidade de registros no período filtrado.",
                top_n=15
            )

            with st.expander("Ver tabela completa de assunto por comunidade"):
                st.dataframe(
                    matriz_assuntos,
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.info("Não há dados suficientes para cruzar assunto e comunidade.")

    with aba_sensiveis:
        st.subheader("Casos sensíveis por período e comunidade")

        casos_sensiveis = f[f["caso_sensivel_flag"] == True].copy()

        if not casos_sensiveis.empty:
            col_sens1, col_sens2 = st.columns(2)

            with col_sens1:
                sensivel_mes = (
                    casos_sensiveis.groupby("ano_mes")
                    .size()
                    .reset_index(name="Casos sensíveis")
                    .sort_values("ano_mes")
                )

                st.plotly_chart(
                    px.line(
                        sensivel_mes,
                        x="ano_mes",
                        y="Casos sensíveis",
                        markers=True,
                        title="Evolução de casos sensíveis por período"
                    ),
                    use_container_width=True
                )

                interpretar(
                    "picos indicam períodos com maior volume de registros sensíveis."
                )

            with col_sens2:
                sensivel_comunidade = (
                    casos_sensiveis.groupby(comunidade_col)
                    .size()
                    .reset_index(name="Casos sensíveis")
                    .rename(columns={comunidade_col: "Comunidade"})
                    .sort_values("Casos sensíveis", ascending=False)
                )

                grafico_ranking_horizontal(
                    sensivel_comunidade,
                    x="Casos sensíveis",
                    y="Comunidade",
                    titulo="Comunidades com mais casos sensíveis",
                    texto="as comunidades no topo concentram maior atenção técnica sensível.",
                    top_n=10
                )

            with st.expander("Ver registros de casos sensíveis"):
                st.dataframe(
                    casos_sensiveis[["data", comunidade_col, "tipo", "tema_principal", "assessor1"]]
                    .rename(columns={
                        comunidade_col: "Comunidade",
                        "tipo": "Tipo",
                        "tema_principal": "Tema",
                        "assessor1": "Assessor"
                    })
                    .sort_values("data", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
        else:
            st.success("Nenhum caso sensível encontrado para os filtros selecionados.")

    with aba_matrizes:
        st.subheader("Tipo de atendimento por tema")

        matriz_tipo_assunto = pd.crosstab(f["tema_principal"], f["tipo"])

        if not matriz_tipo_assunto.empty:
            st.dataframe(
                matriz_tipo_assunto.style.background_gradient(cmap="Blues"),
                use_container_width=True
            )

            interpretar(
                "a matriz mostra quais tipos de atendimento estão mais associados a cada tema."
            )
        else:
            st.info("Não há dados suficientes para cruzar tipo de atendimento e tema.")

        st.markdown("---")

        st.subheader("Tema por comunidade")

        matriz_comunidade_tema = pd.crosstab(f[comunidade_col], f["tema_principal"])

        if not matriz_comunidade_tema.empty:
            st.dataframe(
                matriz_comunidade_tema.style.background_gradient(cmap="Blues"),
                use_container_width=True
            )

            interpretar(
                "esta matriz ajuda a identificar concentração de temas específicos por comunidade."
            )
        else:
            st.info("Não há dados suficientes para cruzar comunidade e tema.")

    with aba_base:
        st.subheader("Comparativo entre base antiga e base atualizada")

        if "base_origem" in f.columns:
            comparativo_base = (
                f.groupby("base_origem")
                .agg(
                    Atendimentos=("tipo", "size"),
                    Participantes=("total_participantes", "sum"),
                    Casos_sensiveis=("caso_sensivel_flag", "sum"),
                    Complexidade_media=("ind_complexidade", "mean")
                )
                .reset_index()
                .rename(columns={
                    "base_origem": "Origem da base",
                    "Casos_sensiveis": "Casos sensíveis",
                    "Complexidade_media": "Complexidade média"
                })
            )

            st.dataframe(
                comparativo_base.style.format({
                    "Participantes": "{:.0f}",
                    "Casos sensíveis": "{:.0f}",
                    "Complexidade média": "{:.1f}"
                }),
                use_container_width=True,
                hide_index=True
            )

            interpretar(
                "compare a origem dos registros para entender diferenças de volume, complexidade e presença de casos sensíveis."
            )

            st.markdown("---")

            grafico_ranking_horizontal(
                comparativo_base,
                x="Atendimentos",
                y="Origem da base",
                titulo="Atendimentos por origem da base",
                texto="mostra o peso de cada base dentro dos filtros selecionados."
            )

            top_assuntos_origem = (
                f.groupby(["base_origem", "tema_principal"])
                .size()
                .reset_index(name="Atendimentos")
                .rename(columns={
                    "base_origem": "Origem da base",
                    "tema_principal": "Tema"
                })
                .sort_values("Atendimentos", ascending=False)
            )

            if not top_assuntos_origem.empty:
                st.plotly_chart(
                    px.bar(
                        top_assuntos_origem.head(20),
                        x="Atendimentos",
                        y="Tema",
                        color="Origem da base",
                        orientation="h",
                        title="Temas mais frequentes por origem da base",
                        text_auto=True
                    ),
                    use_container_width=True
                )

                interpretar(
                    "este gráfico mostra se alguns temas aparecem mais em uma base do que em outra."
                )
        else:
            st.info("A coluna base_origem ainda não está disponível na base carregada.")

# --- 5.11. DASHBOARD PREVISÃO DE DEMANDA ---
elif menu == "🔮 Dashboard Previsão de Demanda":
    st.header("🔮 Previsão de Demanda Técnica")

    historico_mensal = f.groupby("ano_mes").size().reset_index()
    historico_mensal.columns = ["ano_mes", "atendimentos"]
    historico_mensal = historico_mensal.sort_values("ano_mes")
    historico_mensal["data_mes"] = pd.to_datetime(historico_mensal["ano_mes"] + "-01")

    if len(historico_mensal) >= 4:
        historico_mensal["media_movel_3m"] = historico_mensal["atendimentos"].rolling(
            window=3,
            min_periods=1
        ).mean()

        ultimos = historico_mensal.tail(6)

        x = np.arange(len(ultimos))
        y = ultimos["atendimentos"].values
        coef = np.polyfit(x, y, 1)
        tendencia = coef[0]

        ultima_media = historico_mensal["media_movel_3m"].iloc[-1]
        ultimo_mes = historico_mensal["data_mes"].max()

        meses_futuros = pd.date_range(
            start=ultimo_mes + pd.DateOffset(months=1),
            periods=3,
            freq="MS"
        )

        projecoes = []

        for i, mes in enumerate(meses_futuros, start=1):
            valor = max(0, round(ultima_media + (tendencia * i)))

            projecoes.append({
                "data_mes": mes,
                "Período/Mês": mes.strftime("%Y-%m"),
                "Volume de Atendimentos": valor,
                "Tipo": "Projeção"
            })

        historico_plot = historico_mensal.copy()
        historico_plot["Período/Mês"] = historico_plot["ano_mes"]
        historico_plot["Volume de Atendimentos"] = historico_plot["atendimentos"]
        historico_plot["Tipo"] = "Histórico"

        df_plot = pd.concat([
            historico_plot[[
                "data_mes",
                "Período/Mês",
                "Volume de Atendimentos",
                "Tipo"
            ]],
            pd.DataFrame(projecoes)
        ])

        st.plotly_chart(
            px.bar(
                df_plot,
                x="Período/Mês",
                y="Volume de Atendimentos",
                color="Tipo",
                title="Histórico e Projeção dos Próximos 3 Meses",
                color_discrete_sequence=["#1f77b4", "#aec7e8"]
            ),
            use_container_width=True
        )

        if tendencia > 0.5:
            tendencia_txt = "📈 Alta nas Demandas"
        elif tendencia < -0.5:
            tendencia_txt = "📉 Queda nas Demandas"
        else:
            tendencia_txt = "➡️ Operação Estável"

        c1, c2, c3 = st.columns(3)

        c1.metric("Média Móvel Atual", f"{ultima_media:.1f}")
        c2.metric("Tendência da Operação", tendencia_txt)
        c3.metric(
            "Projeção Próximo Mês",
            int(projecoes[0]["Volume de Atendimentos"])
        )

        st.markdown("---")
        st.subheader("💡 Insights Estratégicos de Planejamento")

        pico_historico = historico_mensal["atendimentos"].max()
        mes_pico = historico_mensal.loc[
            historico_mensal["atendimentos"].idxmax(),
            "ano_mes"
        ]

        total_projetado_trimestre = sum([
            p["Volume de Atendimentos"] for p in projecoes
        ])

        col_ins1, col_ins2 = st.columns(2)

        with col_ins1:
            st.info(
                f"📊 **Histórico de Teto Operacional:** O maior pico de atendimentos "
                f"já registrado pela ATI ocorreu em **{mes_pico}**, com um total de "
                f"**{pico_historico}** agendas em um único mês."
            )

            if projecoes[0]["Volume de Atendimentos"] > ultima_media:
                st.warning(
                    "⚠️ **Alerta de Alocação de Recursos:** A projeção matemática aponta "
                    "que o próximo mês ficará **acima da média móvel recente**. Recomenda-se "
                    "que a coordenação evite sobrecarregar as equipes com novos planejamentos "
                    "complexos nas próximas semanas."
                )
            else:
                st.success(
                    "✅ **Estabilização Logística:** O volume previsto para o próximo mês "
                    "indica um ritmo controlado dentro ou abaixo da média histórica, ideal "
                    "para focar na limpeza de relatórios atrasados e organização de drives."
                )

        with col_ins2:
            st.metric(
                "Acumulado Previsto (Próximos 3 Meses)",
                f"{total_projetado_trimestre} Atendimentos",
                help="Soma total das demandas calculadas para o próximo trimestre."
            )

            st.caption(
                "*Nota metodológica:* Este modelo utiliza regressão linear baseada no ritmo "
                "dos últimos 6 meses com peso suavizado pela média móvel trimestral. Fatores "
                "externos e novas ordens de reassentamento podem alterar a precisão real."
            )

    else:
        st.warning(
            "São necessários pelo menos 4 meses de histórico para calcular a linha de tendência e projeções."
        )

# --- 5.11.1. SÉRIES HISTÓRICAS ANUAIS ---
elif menu == "📆 Séries Históricas Anuais":
    st.header("Séries Históricas Anuais")

    st.caption(
        "Compara a evolução anual dos atendimentos, participantes, casos sensíveis, "
        "encaminhamentos e qualidade dos registros."
    )

    aba_resumo, aba_evolucao, aba_indicadores, aba_tabela = st.tabs([
        "Resumo",
        "Evolução",
        "Indicadores",
        "Tabela"
    ])

    serie_anual = (
        f.groupby("ano")
        .agg(
            atendimentos=("tipo", "size"),
            participantes=("total_participantes", "sum"),
            casos_sensiveis=("caso_sensivel_flag", "sum"),
            encaminhamentos=("tem_encaminhamento", "sum"),
            documentos=("doc_flag", "sum"),
            relatorios=("tem_relatorio", "sum"),
            qualidade_media=("ind_qualidade", "mean"),
            complexidade_media=("ind_complexidade", "mean"),
            risco_medio=("score_risco_territorial", "mean")
        )
        .reset_index()
        .sort_values("ano")
    )

    if serie_anual.empty:
        st.info("Não há dados suficientes para gerar séries históricas.")

    else:
        total_atendimentos = int(serie_anual["atendimentos"].sum())
        total_participantes = int(serie_anual["participantes"].sum())
        total_casos_sensiveis = int(serie_anual["casos_sensiveis"].sum())
        qualidade_geral = serie_anual["qualidade_media"].mean()

        ano_inicio = int(serie_anual["ano"].min())
        ano_fim = int(serie_anual["ano"].max())

        with aba_resumo:
            st.subheader("📌 Visão geral da série histórica")

            c1, c2, c3, c4 = st.columns(4)

            c1.metric(
                "Período analisado",
                f"{ano_inicio} a {ano_fim}",
                help="Intervalo de anos disponível na base filtrada."
            )

            c2.metric(
                "Atendimentos",
                total_atendimentos,
                help="Total de atendimentos registrados no período."
            )

            c3.metric(
                "Participantes",
                total_participantes,
                help="Soma de participantes registrados no período."
            )

            c4.metric(
                "Qualidade média",
                f"{qualidade_geral:.1f}%",
                help="Média anual da qualidade documental dos registros."
            )

            interpretar(
                "a série histórica anual permite observar crescimento, redução ou estabilidade dos atendimentos e dos principais indicadores institucionais."
            )

        with aba_evolucao:
            st.subheader("📈 Evolução anual de volume")

            col_anual1, col_anual2 = st.columns(2)

            with col_anual1:
                st.plotly_chart(
                    px.bar(
                        serie_anual,
                        x="ano",
                        y="atendimentos",
                        title="Atendimentos por ano",
                        text_auto=True
                    ),
                    use_container_width=True
                )

            with col_anual2:
                st.plotly_chart(
                    px.line(
                        serie_anual,
                        x="ano",
                        y="participantes",
                        markers=True,
                        title="Participantes alcançados por ano"
                    ),
                    use_container_width=True
                )

            st.caption(
                "A leitura conjunta ajuda a comparar volume de atendimentos com alcance de participantes."
            )

        with aba_indicadores:
            st.subheader("📊 Evolução dos indicadores institucionais")

            serie_long = serie_anual.melt(
                id_vars="ano",
                value_vars=[
                    "casos_sensiveis",
                    "encaminhamentos",
                    "documentos",
                    "relatorios"
                ],
                var_name="Indicador",
                value_name="Quantidade"
            )

            serie_long["Indicador"] = serie_long["Indicador"].replace({
                "casos_sensiveis": "Casos sensíveis",
                "encaminhamentos": "Encaminhamentos",
                "documentos": "Documentos",
                "relatorios": "Relatórios"
            })

            st.plotly_chart(
                px.line(
                    serie_long,
                    x="ano",
                    y="Quantidade",
                    color="Indicador",
                    markers=True,
                    title="Evolução anual dos indicadores institucionais"
                ),
                use_container_width=True
            )

            interpretar(
                "o gráfico mostra como casos sensíveis, encaminhamentos, documentos e relatórios evoluíram ao longo dos anos."
            )

        with aba_tabela:
            st.subheader("📋 Tabela anual consolidada")

            st.caption(
                "Use esta tabela para comparar os principais indicadores ano a ano."
            )

            st.dataframe(
                serie_anual.rename(columns={
                    "ano": "Ano",
                    "atendimentos": "Atendimentos",
                    "participantes": "Participantes",
                    "casos_sensiveis": "Casos sensíveis",
                    "encaminhamentos": "Encaminhamentos",
                    "documentos": "Documentos",
                    "relatorios": "Relatórios",
                    "qualidade_media": "Qualidade média",
                    "complexidade_media": "Complexidade média",
                    "risco_medio": "Risco médio"
                }).style.format({
                    "Participantes": "{:.0f}",
                    "Casos sensíveis": "{:.0f}",
                    "Encaminhamentos": "{:.0f}",
                    "Documentos": "{:.0f}",
                    "Relatórios": "{:.0f}",
                    "Qualidade média": "{:.1f}%",
                    "Complexidade média": "{:.1f}",
                    "Risco médio": "{:.1f}"
                }),
                use_container_width=True,
                hide_index=True
            )

# --- 5.11.2. INDICADORES DE VARIAÇÃO ---
elif menu == "📈 Tendências e Projeções":
    st.header("Tendências e Projeções")

    st.caption(
        "Compara variações recentes, identifica temas emergentes e projeta a demanda dos próximos meses."
    )

    aba_resumo, aba_variacao, aba_temas, aba_projecao, aba_insights = st.tabs([
        "Resumo",
        "Variações",
        "Temas Emergentes",
        "Projeção",
        "Insights"
    ])

    var_mes = calcular_variacao_periodo(f, meses_janela=1)
    var_tri = calcular_variacao_periodo(f, meses_janela=3)
    var_sem = calcular_variacao_periodo(f, meses_janela=6)

    historico_mensal = f.groupby("ano_mes").size().reset_index()
    historico_mensal.columns = ["ano_mes", "atendimentos"]
    historico_mensal = historico_mensal.sort_values("ano_mes")
    historico_mensal["data_mes"] = pd.to_datetime(historico_mensal["ano_mes"] + "-01")

    tem_projecao = len(historico_mensal) >= 4

    if tem_projecao:
        historico_mensal["media_movel_3m"] = (
            historico_mensal["atendimentos"]
            .rolling(window=3, min_periods=1)
            .mean()
        )

        ultimos = historico_mensal.tail(6)

        x = np.arange(len(ultimos))
        y = ultimos["atendimentos"].values
        coef = np.polyfit(x, y, 1)
        tendencia = coef[0]

        ultima_media = historico_mensal["media_movel_3m"].iloc[-1]
        ultimo_mes = historico_mensal["data_mes"].max()

        meses_futuros = pd.date_range(
            start=ultimo_mes + pd.DateOffset(months=1),
            periods=3,
            freq="MS"
        )

        projecoes = []

        for i, mes in enumerate(meses_futuros, start=1):
            valor = max(0, round(ultima_media + (tendencia * i)))

            projecoes.append({
                "data_mes": mes,
                "Período/Mês": mes.strftime("%Y-%m"),
                "Volume de Atendimentos": valor,
                "Tipo": "Projeção"
            })

        historico_plot = historico_mensal.copy()
        historico_plot["Período/Mês"] = historico_plot["ano_mes"]
        historico_plot["Volume de Atendimentos"] = historico_plot["atendimentos"]
        historico_plot["Tipo"] = "Histórico"

        df_plot = pd.concat([
            historico_plot[[
                "data_mes",
                "Período/Mês",
                "Volume de Atendimentos",
                "Tipo"
            ]],
            pd.DataFrame(projecoes)
        ])

        if tendencia > 0.5:
            tendencia_txt = "📈 Alta nas demandas"
        elif tendencia < -0.5:
            tendencia_txt = "📉 Queda nas demandas"
        else:
            tendencia_txt = "➡️ Operação estável"

        pico_historico = historico_mensal["atendimentos"].max()
        mes_pico = historico_mensal.loc[
            historico_mensal["atendimentos"].idxmax(),
            "ano_mes"
        ]

        total_projetado_trimestre = sum([
            p["Volume de Atendimentos"] for p in projecoes
        ])

    with aba_resumo:
        st.subheader("📌 Visão geral das tendências")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Variação mensal",
            f"{var_mes['atual']} atendimentos",
            delta=var_mes["texto"],
            help="Compara o mês mais recente com o mês anterior."
        )

        c2.metric(
            "Variação trimestral",
            f"{var_tri['atual']} atendimentos",
            delta=var_tri["texto"],
            help="Compara os últimos 3 meses com os 3 meses anteriores."
        )

        c3.metric(
            "Variação semestral",
            f"{var_sem['atual']} atendimentos",
            delta=var_sem["texto"],
            help="Compara os últimos 6 meses com os 6 meses anteriores."
        )

        if tem_projecao:
            c4.metric(
                "Próximo mês",
                int(projecoes[0]["Volume de Atendimentos"]),
                help="Estimativa calculada para o próximo mês."
            )
        else:
            c4.metric(
                "Próximo mês",
                "Sem base",
                help="São necessários pelo menos 4 meses de histórico."
            )

        interpretar(
            "este painel reúne a leitura do comportamento recente da demanda e a projeção dos próximos meses."
        )

    with aba_variacao:
        st.subheader("📈 Evolução mensal de atendimentos")

        evolucao = (
            f.groupby("ano_mes")
            .size()
            .reset_index(name="Atendimentos")
            .sort_values("ano_mes")
        )

        st.plotly_chart(
            px.line(
                evolucao,
                x="ano_mes",
                y="Atendimentos",
                markers=True,
                title="Atendimentos por mês"
            ),
            use_container_width=True
        )

        st.info(
            "Como interpretar: valores positivos indicam aumento de demanda. "
            "Valores negativos indicam redução. Quando aparece 'Sem base anterior', "
            "não existe período anterior suficiente para comparação."
        )

    with aba_temas:
        st.subheader("🚨 Temas emergentes")

        riscos = top_riscos_emergentes(
            f,
            coluna_tema="tema_principal",
            meses_janela=3,
            top_n=10
        )

        if not riscos.empty:
            riscos_exibir = riscos.rename(columns={
                "tema_principal": "Tema",
                "periodo_atual": "Últimos 3 meses",
                "periodo_anterior": "3 meses anteriores",
                "crescimento_abs": "Crescimento absoluto",
                "crescimento_pct": "Crescimento percentual"
            })

            st.dataframe(
                riscos_exibir.style.format({
                    "Crescimento percentual": "{:.1f}%"
                }),
                use_container_width=True,
                hide_index=True
            )

            st.warning(
                "Leitura: esses temas merecem atenção porque cresceram no período recente. "
                "Quando o crescimento percentual estiver vazio, significa que o tema apareceu agora "
                "ou não tinha base anterior."
            )

        else:
            st.success(
                "Nenhum tema emergente identificado com base suficiente no período filtrado."
            )

    with aba_projecao:
        st.subheader("🔮 Projeção de demanda")

        if tem_projecao:
            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Média móvel atual",
                f"{ultima_media:.1f}",
                help="Média dos últimos 3 meses de atendimento."
            )

            c2.metric(
                "Tendência",
                tendencia_txt,
                help="Leitura da variação recente da demanda."
            )

            c3.metric(
                "Próximos 3 meses",
                int(total_projetado_trimestre),
                help="Soma das projeções para o próximo trimestre."
            )

            fig = px.bar(
                df_plot,
                x="Período/Mês",
                y="Volume de Atendimentos",
                color="Tipo",
                title="Histórico e Projeção dos Próximos 3 Meses",
                color_discrete_sequence=["#1f77b4", "#aec7e8"]
            )

            st.plotly_chart(fig, use_container_width=True)

            st.caption(
                "A projeção combina média móvel trimestral com tendência observada nos últimos 6 meses."
            )

        else:
            st.warning(
                "São necessários pelo menos 4 meses de histórico para calcular tendência e projeções."
            )

    with aba_insights:
        st.subheader("💡 Insights Estratégicos de Planejamento")

        if tem_projecao:
            col_ins1, col_ins2 = st.columns(2)

            with col_ins1:
                st.info(
                    f"📊 **Teto operacional:** o maior pico registrado ocorreu em "
                    f"**{mes_pico}**, com **{pico_historico}** atendimentos em um único mês."
                )

                if projecoes[0]["Volume de Atendimentos"] > ultima_media:
                    st.warning(
                        "⚠️ **Alerta de alocação:** a projeção indica volume acima da média "
                        "móvel recente. Vale evitar sobrecarga da equipe nas próximas semanas."
                    )
                else:
                    st.success(
                        "✅ **Cenário controlado:** o volume previsto está dentro ou abaixo "
                        "da média recente, favorecendo organização de relatórios e registros."
                    )

            with col_ins2:
                st.metric(
                    "Previsão do trimestre",
                    f"{total_projetado_trimestre} atendimentos",
                    help="Soma total das demandas calculadas para os próximos 3 meses."
                )

                st.caption(
                    "Nota metodológica: este modelo usa regressão linear dos últimos 6 meses "
                    "com suavização pela média móvel trimestral."
                )

        else:
            st.info(
                "Ainda não há histórico suficiente para gerar insights preditivos com segurança."
            )

# --- 5.11.3. METAS E RELATORIO EXECUTIVO---
elif menu == "🎯 Metas e Relatório Executivo":
    st.header("Metas e Relatório Executivo")

    st.caption(
        "Acompanha metas de qualidade, evidências, encaminhamentos e gera uma leitura executiva do período."
    )

    aba_resumo, aba_metas, aba_relatorio = st.tabs([
        "Resumo",
        "Metas",
        "Relatório"
    ])

    meta_qualidade = 80
    meta_drive = 85
    meta_relatorio = 80
    meta_casos_sensiveis = 15
    meta_encaminhamento = 40

    qualidade_media = f["ind_qualidade"].mean()
    perc_drive = f["drive_flag"].mean() * 100
    perc_relatorio = f["tem_relatorio"].mean() * 100
    perc_sensiveis = f["caso_sensivel_flag"].mean() * 100
    perc_encaminhamento = f["tem_encaminhamento"].mean() * 100

    metas = pd.DataFrame({
        "Indicador": [
            "Qualidade média",
            "Registros salvos no drive",
            "Registros com relatório",
            "Percentual de casos sensíveis",
            "Registros com encaminhamento"
        ],
        "Resultado": [
            qualidade_media,
            perc_drive,
            perc_relatorio,
            perc_sensiveis,
            perc_encaminhamento
        ],
        "Meta": [
            meta_qualidade,
            meta_drive,
            meta_relatorio,
            meta_casos_sensiveis,
            meta_encaminhamento
        ],
        "Tipo": [
            "mínimo",
            "mínimo",
            "mínimo",
            "máximo",
            "mínimo"
        ]
    })

    metas["Status"] = np.where(
        (
            ((metas["Tipo"] == "mínimo") & (metas["Resultado"] >= metas["Meta"])) |
            ((metas["Tipo"] == "máximo") & (metas["Resultado"] <= metas["Meta"]))
        ),
        "Dentro da meta",
        "Fora da meta"
    )

    metas["Distância da meta"] = np.where(
        metas["Tipo"] == "mínimo",
        metas["Resultado"] - metas["Meta"],
        metas["Meta"] - metas["Resultado"]
    )

    metas["Resultado visual"] = metas["Resultado"].round(1)
    metas["Meta visual"] = metas["Meta"].round(1)

    dentro_meta = int((metas["Status"] == "Dentro da meta").sum())
    fora_meta = int((metas["Status"] == "Fora da meta").sum())

    with aba_resumo:
        st.subheader("📌 Visão geral das metas")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Indicadores dentro da meta",
            dentro_meta,
            help="Quantidade de indicadores que atingiram o critério esperado."
        )

        c2.metric(
            "Indicadores fora da meta",
            fora_meta,
            help="Quantidade de indicadores que precisam de atenção."
        )

        c3.metric(
            "Qualidade média",
            f"{qualidade_media:.1f}%",
            help="Percentual médio de qualidade documental dos registros filtrados."
        )

        c4.metric(
            "Casos sensíveis",
            f"{perc_sensiveis:.1f}%",
            help="Percentual de registros classificados como caso sensível."
        )

        interpretar(
            "qualidade, drive, relatório e encaminhamento devem ficar acima da meta. Casos sensíveis usam limite máximo, pois indicam necessidade de atenção técnica."
        )

        st.markdown("---")

        status_metas = (
            metas["Status"]
            .value_counts()
            .reset_index()
        )

        status_metas.columns = ["Status", "Indicadores"]

        grafico_ranking_horizontal(
            status_metas,
            x="Indicadores",
            y="Status",
            titulo="Status geral das metas",
            texto="mostra quantos indicadores estão dentro ou fora do esperado."
        )

    with aba_metas:
        st.subheader("🎯 Resultado por indicador")

        metas_grafico = metas.copy()
        metas_grafico["Indicador"] = metas_grafico["Indicador"].astype(str)

        grafico_ranking_horizontal(
            metas_grafico,
            x="Resultado visual",
            y="Indicador",
            titulo="Resultado atual dos indicadores",
            texto="compare o resultado atual com a meta na tabela abaixo."
        )

        st.markdown("---")

        st.subheader("📋 Tabela de acompanhamento das metas")

        st.dataframe(
            metas[[
                "Indicador",
                "Resultado",
                "Meta",
                "Tipo",
                "Distância da meta",
                "Status"
            ]].style.format({
                "Resultado": "{:.1f}%",
                "Meta": "{:.1f}%",
                "Distância da meta": "{:.1f} p.p."
            }),
            use_container_width=True,
            hide_index=True
        )

        interpretar(
            "distância positiva indica folga em relação à meta. Distância negativa indica quanto falta para atingir o esperado."
        )

    with aba_relatorio:
        st.subheader("📝 Resumo executivo automático")

        st.caption(
            "Texto pronto para copiar, revisar e usar em relatório, apresentação ou prestação de contas."
        )

        resumo = gerar_resumo_executivo(f)

        st.text_area(
            "Texto do relatório",
            resumo,
            height=420
        )

        st.download_button(
            label="Baixar resumo executivo em TXT",
            data=resumo.encode("utf-8"),
            file_name="resumo_executivo_ati.txt",
            mime="text/plain"
        )

# --- 5.11.5. METODOLOGIA E DICIONÁRIO DE DADOS---
elif menu == "ℹ️ Metodologia e Dicionário de Dados":
    st.header("ℹMetodologia e Dicionário de Dados")

    st.caption(
        "Explica como os indicadores são calculados e como interpretar os resultados do sistema."
    )

    st.subheader("📘 Como ler o painel")

    st.info(
        "O sistema transforma os registros de assessoramento em indicadores de volume, público atendido, "
        "território, tema, risco, reincidência, carga de equipe e qualidade das evidências."
    )

    st.markdown("""
    **Atendimentos**  
    Quantidade de registros válidos no período filtrado.

    **Participantes**  
    Soma das pessoas informadas por sexo e faixa etária.

    **Casos sensíveis**  
    Registros marcados como caso sensível. Indicam maior atenção técnica.

    **Encaminhamentos**  
    Registros que possuem texto de encaminhamento preenchido.

    **Qualidade média**  
    Mede se o atendimento tem documento, relatório e registro no drive.

    **Score de risco territorial**  
    Índice de 0 a 100 calculado com base em vulnerabilidade, reincidência e casos sensíveis.

    **Reincidência**  
    Indica repetição de temas e demandas em comunidades ou territórios.

    **Sobrecarga da equipe**  
    Soma duração, complexidade e peso adicional de casos sensíveis.
    """)

    st.markdown("---")

    st.subheader("📋 Dicionário das principais colunas")

    dicionario = pd.DataFrame([
        {"Campo": "data", "Descrição": "Data do atendimento registrado."},
        {"Campo": "tipo", "Descrição": "Tipo de agenda ou atendimento."},
        {"Campo": "assessor1", "Descrição": "Principal assessor responsável."},
        {"Campo": "assessor2", "Descrição": "Segundo assessor, quando houver."},
        {"Campo": "comunidade_analise", "Descrição": "Comunidade usada para análise territorial."},
        {"Campo": "tema_principal", "Descrição": "Tema consolidado do atendimento."},
        {"Campo": "total_participantes", "Descrição": "Total de pessoas participantes."},
        {"Campo": "caso_sensivel_flag", "Descrição": "Indica se o registro é caso sensível."},
        {"Campo": "tem_encaminhamento", "Descrição": "Indica se há encaminhamento registrado."},
        {"Campo": "ind_complexidade", "Descrição": "Índice de complexidade do atendimento."},
        {"Campo": "ind_sobrecarga", "Descrição": "Índice de carga técnica da equipe."},
        {"Campo": "ind_reincidencia", "Descrição": "Frequência combinada de tema e território."},
        {"Campo": "ind_vulnerabilidade", "Descrição": "Índice de vulnerabilidade associado ao atendimento."},
        {"Campo": "ind_qualidade", "Descrição": "Percentual de qualidade documental do registro."},
        {"Campo": "score_risco_territorial", "Descrição": "Score consolidado de risco territorial, de 0 a 100."}
    ])

    st.dataframe(
        dicionario,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.subheader("⚠️ Observações metodológicas")

    st.warning(
        "O mapa territorial atual usa coordenadas simuladas. Ele serve apenas como visual analítico, "
        "não como mapa geográfico real. Para uso oficial, é necessário incluir latitude e longitude reais das comunidades."
    )

    st.info(
        "A análise utiliza os identificadores dos assessoramentos e dos núcleos familiares quando disponíveis."
    )
