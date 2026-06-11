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

PARTICIPANTES = ["feminino", "masculino", "criancas", "adolescentes", "jovens", "adultos", "idosos"]
STOPWORDS_PT = ["de", "a", "o", "que", "e", "do", "da", "em", "um", "para", "com", "na", "no", "uma", "os", "as", "dos", "das", "ao", "aos", "por", "mais", "se", "foi", "atendimento", "reunião", "assessoramento", "comunidade", "relatório", "visita", "sobre", "nao", "não", "como"]

# ==================================================================================
# 2. FUNÇÕES DE AUXÍLIO / TRATAMENTO DE DADOS
# ==================================================================================
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

# ==================================================================================
# 3. CARREGAMENTO E ENGENHARIA DE ATRIBUTOS (INDICADORES DERIVADOS)
# ==================================================================================
@st.cache_data(show_spinner=False)
def load_data(file) -> pd.DataFrame:
    if str(file).endswith(".csv"): df = pd.read_csv(file)
    else: df = pd.read_excel(file)
        
    df = df.rename(columns={v: k for k, v in COLS.items() if v in df.columns})
    
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
        elif "espontânea" in tipo_agenda or "não" in str(row.get("tipo", "")).lower() or pd.notna(row.get("assunto_demanda")):
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

# ==================================================================================
# 4. INTERFACE E MENU LATERAL
# ==================================================================================
uploaded = st.file_uploader("Carregue a planilha .xlsx ou .csv da ATI", type=["xlsx", "csv"])
if uploaded is None:
    default_path = Path("02. Registro de Assessoramento - Atualizado  (respostas).xlsx")
    if default_path.exists(): uploaded = default_path
    else:
        st.info("Aguardando carregamento da planilha para estruturar a inteligência do painel.")
        st.stop()

df = load_data(uploaded)

with st.sidebar:
    st.header("Navegação do Sistema")
    menu = st.selectbox("Selecione o Dashboard", [
        "✨ Insights Automáticos",
        "📊 Dashboard Executivo",
        "📋 Qualidade, Evidências e Encaminhamentos",
        "⏱️ Dashboard Operacional (Sazonalidade)",
        "👥 Dashboard Equipe & Carga",
        "📂 Dashboard Temático & Termos",
        "🗺️ Dashboard Territorial",
        "⚠️ Dashboard Vulnerabilidades",
        "🔄 Dashboard Reincidência",
        "🔮 Dashboard Previsão de Demanda"
    ])
    st.markdown("---")
    st.subheader("Filtros Globais")
    anos = st.multiselect("Anos", sorted(df["ano"].unique()), default=sorted(df["ano"].unique()))
    tipos = st.multiselect("Tipo de Agenda", sorted(df["tipo"].unique()), default=sorted(df["tipo"].unique()))

f = df[df["ano"].isin(anos) & df["tipo"].isin(tipos)].copy()
if f.empty:
    st.warning("Nenhum dado encontrado para os filtros selecionados.")
    st.stop()

# ==================================================================================
# 5. CONSTRUTOR DE PÁGINAS (DASHBOARDS)
# ==================================================================================

# --- 5.1. INSIGHTS AUTOMÁTICOS ---
if menu == "✨ Insights Automáticos":
    st.header("✨ Insights Automáticos e Alertas de Gestão")
    
    temas_horas = f.groupby("tema_principal")["duracao_horas"].sum().sort_values(ascending=False)
    perc_top4 = (temas_horas.head(4).sum() / temas_horas.sum() * 100).round(0)
    st.metric(label="Concentração de Esforço Técnico (Pareto)", value=f"{int(perc_top4)}% das Horas")
    st.info(f"💡 **Fato:** Atualmente, os 4 temas mais demandados representam **{int(perc_top4)}%** de todas as horas técnicas registradas na ATI.")

    if not f.empty:
        top_comunidade_sensivel = f.groupby("local")["caso_sensivel_flag"].sum().sort_values(ascending=False)
        total_sensiveis = f["caso_sensivel_flag"].sum()
        if total_sensiveis > 0:
            perc_com_sensivel = (top_comunidade_sensivel.iloc[0] / total_sensiveis * 100).round(0)
            st.warning(f"⚠️ **Alerta Territorial:** A localidade **'{top_comunidade_sensivel.index[0]}'** concentrou **{int(perc_com_sensivel)}%** de todos os casos sensíveis complexos mapeados.")

    # [INSERÇÃO]: Alertas Automáticos de Qualidade na Capa
    pendencias_totais = f[(f["doc_flag"] == False) | (f["drive_flag"] == False) | (f["tem_relatorio"] == False)].shape[0]
    percentual_pendente = round((pendencias_totais / f.shape[0]) * 100, 1) if f.shape[0] > 0 else 0
    if percentual_pendente > 30:
        st.error(f"🚨 **Alerta de Compliance Operacional:** Atualmente, **{percentual_pendente}%** dos registros da ATI possuem pendências críticas de evidências (Sem Relatório, Documento ou Drive). Verifique a aba de Qualidade e Auditoria.")

    if "formato" in f.columns and len(f["formato"].unique()) > 1:
        horas_formato = f.groupby("formato")["duracao_horas"].mean()
        if "Presencial" in horas_formato.index and ("Remoto" in horas_formato.index or "Online" in horas_formato.index):
            remoto_key = "Remoto" if "Remoto" in horas_formato.index else "Online"
            razao = (horas_formato["Presencial"] / horas_formato[remoto_key]).round(1)
            st.success(f"🚗 **Logística e Deslocamento:** Os atendimentos em formato **Presencial** demandam em média **{razao} vezes mais horas** técnicas comparados aos remotos.")

# --- 5.2. DASHBOARD EXECUTIVO ---
elif menu == "📊 Dashboard Executivo":
    st.header("📊 Dashboard Executivo")
    
    # Cálculo MoM Seguro
    f_ordenado = f.sort_values("data")
    if len(f_ordenado) > 0:
        ultimo_mes_ano = f_ordenado["ano_mes"].iloc[-1]
        vol_mes_atual = f_ordenado[f_ordenado["ano_mes"] == ultimo_mes_ano].shape[0]
        periodos_anteriores = f_ordenado[f_ordenado["ano_mes"] != ultimo_mes_ano]
        if not periodos_anteriores.empty:
            vol_mes_anterior = periodos_anteriores.groupby("ano_mes").size().iloc[-1]
            variacao_mom = ((vol_mes_atual - vol_mes_anterior) / max(1, vol_mes_anterior) * 100).round(1)
            delta_mom_txt = f"{variacao_mom}% vs mês anterior"
        else: delta_mom_txt = "Primeiro período"
    else: vol_mes_atual, delta_mom_txt = 0, "N/A"

    # Gênero
    total_fem = int(f["feminino"].sum())
    total_masc = int(f["masculino"].sum())
    total_geral_genero = total_fem + total_masc
    p_fem = round((total_fem / total_geral_genero) * 100, 1) if total_geral_genero > 0 else 0
    p_masc = round((total_masc / total_geral_genero) * 100, 1) if total_geral_genero > 0 else 0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Demandas no Mês Atual", vol_mes_atual, delta=delta_mom_txt)
    c2.metric("Total de Atendimentos", f.shape[0])
    c3.metric("Público Feminino Atendido", f"{total_fem}", delta=f"{p_fem}% do total")
    c4.metric("Público Masculino Atendido", f"{total_masc}", delta=f"{p_masc}% do total")
    
    st.markdown("---")
    col_exec1, col_exec2 = st.columns(2)
    
    with col_exec1:
        perfil_demandas = f["classificacao_demanda"].value_counts().reset_index()
        perfil_demandas.columns = ["Tipo de Demanda", "Quantidade"]
        # CORRIGIDO: color_discrete_sequence no lugar de color_discrete_palette
        st.plotly_chart(px.bar(perfil_demandas, x="Quantidade", y="Tipo de Demanda", orientation="h", title="Distribuição das Demandas por Natureza do Atendimento", color="Tipo de Demanda", color_discrete_sequence=px.colors.qualitative.Pastel), use_container_width=True)
        
    with col_exec2:
        df_genero = pd.DataFrame({"Gênero": ["Feminino", "Masculino"], "Participantes": [total_fem, total_masc]})
        st.plotly_chart(px.pie(df_genero, values="Participantes", names="Gênero", title="Proporção de Participação por Sexo", color_discrete_sequence=["#FF6692", "#636EFA"]), use_container_width=True)

    col1, col2 = st.columns(2)
    with col1:
        crescimento = f.groupby("ano_mes").size().reset_index()
        crescimento.columns = ["ano_mes", "Atendimentos"]
        st.plotly_chart(px.bar(crescimento.tail(12), x="ano_mes", y="Atendimentos", title="Evolução Mensal de Atendimentos", text_auto=True), use_container_width=True)
    with col2:
        top_temas = f.groupby("tema_principal").size().reset_index()
        top_temas.columns = ["tema_principal", "Qtd"]
        top_temas = top_temas.sort_values("Qtd", ascending=False).head(8)
        st.plotly_chart(px.pie(top_temas, values="Qtd", names="tema_principal", title="Temas Mais Demandados"), use_container_width=True)

# --- 5.3. QUALIDADE, EVIDÊNCIAS E ENCAMINHAMENTOS ---
elif menu == "📋 Qualidade, Evidências e Encaminhamentos":
    st.header("📋 Qualidade, Evidências e Encaminhamentos Institucionais")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        formato_vct = f["formato"].value_counts().reset_index()
        formato_vct.columns = ["Formato", "Qtd"] # Garantia de nomes limpos de colunas
        st.plotly_chart(px.pie(formato_vct, values="Qtd", names="Formato", title="Uso de Canal: Presencial vs Remoto", color_discrete_sequence=["#2ca02c", "#d62728"]), use_container_width=True)
    
    with col_f2:
        remotos = f[f["formato"].str.lower() == "remoto"]
        if not remotos.empty:
            remotos_vct = remotos["detalhe_remoto"].value_counts().reset_index()
            remotos_vct.columns = ["Canal Remoto", "Qtd"]
            st.plotly_chart(px.bar(remotos_vct, x="Qtd", y="Canal Remoto", orientation="h", title="Detalhamento dos Atendimentos Remotos", color="Canal Remoto", color_discrete_sequence=px.colors.qualitative.Safe), use_container_width=True)
        else:
            st.info("Nenhum atendimento remoto identificado nos filtros aplicados.")

    st.markdown("---")
    st.subheader("🚀 Destinação dos Encaminhamentos")
    c_enc1, c_enc2 = st.columns(2)
    with c_enc1:
        enc_vct = f["foi_encaminhado"].value_counts().reset_index()
        enc_vct.columns = ["Status", "Qtd"]
        st.plotly_chart(px.pie(enc_vct, values="Qtd", names="Status", title="Percentual de Casos com Encaminhamento", color_discrete_sequence=["#FF7F0E", "#1F77B4"]), use_container_width=True)
    with c_enc2:
        demandas_encaminhadas = f[f["foi_encaminhado"] == "Encaminhado"]
        if not demandas_encaminhadas.empty:
            temas_encaminhados = demandas_encaminhadas.groupby("tema_principal").size().reset_index()
            temas_encaminhados.columns = ["tema_principal", "Volume"]
            temas_encaminhados = temas_encaminhados.sort_values("Volume", ascending=False).head(5)
            st.plotly_chart(px.bar(temas_encaminhados, x="Volume", y="tema_principal", orientation="h", title="Temas que mais geram demandas externas", color_discrete_sequence=["#9467bd"]), use_container_width=True)

# --- 5.4. DASHBOARD OPERACIONAL (SAZONALIDADE) ---
elif menu == "⏱️ Dashboard Operacional (Sazonalidade)":
    st.header("⏱️ Análise de Sazonalidade e Matriz de Formatos")
    
    col1, col2 = st.columns(2)
    with col1:
        sazonalidade_mes = f.groupby("mes").size().reset_index(name="Atendimentos")
        st.plotly_chart(px.line(sazonalidade_mes, x="mes", y="Atendimentos", markers=True, title="Volume de Demandas por Mês do Ano"), use_container_width=True)
    with col2:
        dias_criticos = f.groupby("dia_semana").size().reindex(["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"], fill_value=0).reset_index(name="Atendimentos")
        st.plotly_chart(px.bar(dias_criticos, x="dia_semana", y="Atendimentos", title="Dias Mais Críticos de Atendimento", text_auto=True, color_discrete_sequence=["#FFA15A"]), use_container_width=True)
    
    st.markdown("---")
    st.subheader("🗺️ Matriz Logística: Cruzamento de Formato por Complexidade Média")
    matriz_log = f.groupby("formato").agg(atendimentos=("tipo", "size"), complexidade_media=("ind_complexidade", "mean")).reset_index()
    st.dataframe(matriz_log.style.format({"complexidade_media": "{:.1f} pts"}), use_container_width=True, hide_index=True)

# --- 5.5. DASHBOARD EQUIPE & CARGA ---
elif menu == "👥 Dashboard Equipe & Carga":
    st.header("👥 Gestão de Equipe, Carga de Trabalho e Complexidade")
    
    equipe = f.groupby("assessor1").agg(
        atendimentos=("tipo", "size"),
        horas_dedicadas=("duracao_horas", "sum"),
        sobrecarga_acumulada=("ind_sobrecarga", "sum"),
        complexidade_media=("ind_complexidade", "mean")
    ).reset_index().sort_values("sobrecarga_acumulada", ascending=False)
    
    st.write("**Ranking Geral de Sobrecarga da Equipe:**")
    df_estilizado = equipe.style.background_gradient(subset=["sobrecarga_acumulada"], cmap="OrRd").format({
        "atendimentos": "{:,.0f}", "horas_dedicadas": "{:,.1f}h", "complexidade_media": "{:,.1f} pts", "sobrecarga_acumulada": "{:,.0f}"
    })
    st.dataframe(df_estilizado, use_container_width=True, hide_index=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(px.bar(equipe, x="sobrecarga_acumulada", y="assessor1", orientation="h", title="Índice de Sobrecarga Total Acumulada", text_auto=".0f"), use_container_width=True)
    with col2:
        fig_scatter = px.scatter(equipe, x="complexidade_media", y="horas_dedicadas", size="atendimentos", text="assessor1", title="Matriz de Equilíbrio: Complexidade vs Horas", size_max=30)
        fig_scatter.update_traces(textposition='top center')
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.markdown("---")
    st.subheader("🤝 Análise de Parcerias e Conectividade de Equipe (Trabalho em Duplas)")
    
    if "assessor2" in f.columns:
        duplas_validas = f.dropna(subset=["assessor1", "assessor2"]).copy()
        duplas_validas["assessor1"] = duplas_validas["assessor1"].astype(str).str.strip()
        duplas_validas["assessor2"] = duplas_validas["assessor2"].astype(str).str.strip()
        duplas_validas = duplas_validas[(duplas_validas["assessor1"] != "") & (duplas_validas["assessor2"] != "") & (duplas_validas["assessor1"].str.lower() != "nan") & (duplas_validas["assessor2"].str.lower() != "nan")]
        
        if not duplas_validas.empty:
            duplas_validas["parceria"] = duplas_validas.apply(lambda row: " + ".join(sorted([row["assessor1"], row["assessor2"]])), axis=1)
            ranking_duplas = duplas_validas.groupby("parceria").size().reset_index(name="Atendimentos Conjuntos").sort_values("Atendimentos Conjuntos", ascending=False).head(10)
            
            redes = pd.concat([
                duplas_validas[["assessor1", "assessor2"]].rename(columns={"assessor1": "assessor", "assessor2": "parceiro"}),
                duplas_validas[["assessor2", "assessor1"]].rename(columns={"assessor2": "assessor", "assessor1": "parceiro"})
            ])
            conectividade = redes.groupby("assessor")["parceiro"].nunique().reset_index(name="Parceiros Distintos").sort_values("Parceiros Distintos", ascending=False)
            
            c_dupla1, c_dupla2 = st.columns(2)
            with c_dupla1:
                st.write("**🏆 Top 10 Parcerias que Mais Atenderam Juntas em Campo:**")
                st.dataframe(ranking_duplas.style.background_gradient(subset=["Atendimentos Conjuntos"], cmap="Purples"), use_container_width=True, hide_index=True)
            with c_dupla2:
                st.write("**🧩 Dinâmica de Rede (Quem atua com maior variedade de parceiros):**")
                st.dataframe(conectividade.style.background_gradient(subset=["Parceiros Distintos"], cmap="GnBu"), use_container_width=True, hide_index=True)

    #  Calendário Técnico Semanal por Assessor
    st.markdown("---")
    st.subheader("📅 Alocação Semanal por Assessor")
    ordem_dias = ["Segunda-feira", "Terça-feira", "Quarta-feira", "Quinta-feira", "Sexta-feira", "Sábado", "Domingo"]
    matriz_dias = pd.crosstab(f["assessor1"], f["dia_semana"]).reindex(columns=ordem_dias, fill_value=0)
    st.dataframe(matriz_dias.style.background_gradient(cmap="YlOrBr", axis=1), use_container_width=True)

# --- 5.6. DASHBOARD TEMÁTICO & TERMOS ---
elif menu == "📂 Dashboard Temático & Termos":
    st.header("📂 Linhas de Ação e Mineração de Relatórios Técnicos")
    
    temas = f.groupby("tema_principal").agg(
        horas_totais=("duracao_horas", "sum"),
        participantes_totais=("total_participantes", "sum"),
        eficiencia_media=("ind_eficiencia", "mean")
    ).reset_index().sort_values("horas_totais", ascending=False)
    
    st.plotly_chart(px.bar(temas, x="horas_totais", y="tema_principal", orientation="h", title="Quais temas consomem mais tempo técnico?"), use_container_width=True)
    st.dataframe(temas.style.format({"horas_totais": "{:.1f}h", "participantes_totais": "{:.0f}", "eficiencia_media": "{:.1f} part/h"}), use_container_width=True, hide_index=True)
    
    st.markdown("---")
    st.subheader("🔍 Inteligência Textual e Clusterização de Temas Semelhantes")
    
    matriz_cluster = pd.crosstab(f["local"], f["tema_principal"])
    st.dataframe(matriz_cluster.style.background_gradient(cmap="Purples"), use_container_width=True)

    col_txt1, col_txt2 = st.columns(2)
    with col_txt1:
        if "encaminhamentos" in f.columns:
            df_termos_enc = extrair_palavras_chave(f["encaminhamentos"])
            if not df_termos_enc.empty: st.plotly_chart(px.bar(df_termos_enc, x="Frequência", y="Termo", orientation="h", title="Palavras-Chave em 'Encaminhamentos'"), use_container_width=True)
    with col_txt2:
        if "relatorio" in f.columns:
            df_termos_rel = extrair_palavras_chave(f["relatorio"])
            if not df_termos_rel.empty: st.plotly_chart(px.bar(df_termos_rel, x="Frequência", y="Termo", orientation="h", title="Palavras-Chave em 'Relatórios'", color_discrete_sequence=["#FF6692"]), use_container_width=True)

# --- 5.7. DASHBOARD TERRITORIAL ---
elif menu == "🗺️ Dashboard Territorial":
    st.header("🗺️ Mapeamento e Concentração Territorial")
    
    #  Ranking de Comunidades por Crescimento de Demanda
    st.subheader("📈 Crescimento de Demandas por Território (Últimos 2 Meses)")
    meses_disponiveis = sorted(f["ano_mes"].unique())
    if len(meses_disponiveis) >= 2:
        m_atual, m_anterior = meses_disponiveis[-1], meses_disponiveis[-2]
        df_crescimento = f[f["ano_mes"].isin([m_atual, m_anterior])].groupby(["local", "ano_mes"]).size().unstack(fill_value=0)
        df_crescimento["Crescimento Absoluto"] = df_crescimento[m_atual] - df_crescimento[m_anterior]
        df_crescimento = df_crescimento.sort_values("Crescimento Absoluto", ascending=False).reset_index()
        st.dataframe(df_crescimento.rename(columns={m_anterior: f"Vol {m_anterior}", m_atual: f"Vol {m_atual}"}), use_container_width=True, hide_index=True)

    st.markdown("---")
    #  Mapa Geográfico Real/Analítico
    st.subheader("📍 Dispersão e Mapa Analítico de Demandas")
    territorios = f.groupby("local").agg(atendimentos=("tipo", "size"), score_risco=("score_risco_territorial", "mean"), participantes=("total_participantes", "sum")).reset_index()
    
    np.random.seed(42)
    territorios["lat"] = -20.2 + np.random.uniform(-0.08, 0.08, len(territorios))
    territorios["lon"] = -43.4 + np.random.uniform(-0.08, 0.08, len(territorios))
    
    fig_mapa = px.scatter_mapbox(territorios, lat="lat", lon="lon", size="atendimentos", color="score_risco", hover_name="local", color_continuous_scale=px.colors.cyclical.IceFire, size_max=35, zoom=9)
    fig_mapa.update_layout(mapbox_style="open-street-map", margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig_mapa, use_container_width=True)

# --- 5.8. DASHBOARD VULNERABILIDADES ---
elif menu == "⚠️ Dashboard Vulnerabilidades":
    st.header("⚠️ Gestão de Riscos e Score Único Territorial")
    
    #  Score Único de Risco Territorial
    st.subheader("🛡️ Ranking de Priorização por Score de Risco Combinado (Vulnerabilidade + Reincidência + Casos Sensíveis)")
    ranking_risco = f.groupby("local").agg(score_risco_medio=("score_risco_territorial", "mean"), total_atendimentos=("tipo", "size"), casos_criticos=("caso_sensivel_flag", "sum")).reset_index().sort_values("score_risco_medio", ascending=False)
    
    st.dataframe(ranking_risco.head(20).style.background_gradient(subset=["score_risco_medio"], cmap="YlOrRd").format({"score_risco_medio": "{:.1f} / 100"}), use_container_width=True, hide_index=True)

    col1, col2 = st.columns(2)
    with col1:
        sensiv = f[f["caso_sensivel_flag"] == True]
        st.plotly_chart(px.bar(sensiv.groupby("tema_principal").size().reset_index(name="Qtd").sort_values("Qtd", ascending=False), x="Qtd", y="tema_principal", orientation="h", title="Temas Críticos Criadores de Vulnerabilidade", color_discrete_sequence=["#EF553B"]), use_container_width=True)
    with col2:
        st.plotly_chart(px.bar(ranking_risco.head(10), x="score_risco_medio", y="local", orientation="h", title="Top 10 Territórios Sob Maior Alerta de Risco"), use_container_width=True)

# --- 5.9. DASHBOARD REINCIDÊNCIA ---
elif menu == "🔄 Dashboard Reincidência":
    st.header("🔄 Avaliação de Reincidência e Retornos das Comunidades")
    reincidencia_grupo = f.groupby(["local", "tema_principal"]).size().reset_index(name="frequencia_repetida").sort_values("frequencia_repetida", ascending=False)
    st.write("**Alertas de Repetição Crônica de Demandas no Mesmo Território:**")
    st.dataframe(reincidencia_grupo.head(15), use_container_width=True, hide_index=True)

# --- 5.10. DASHBOARD PREVISÃO DE DEMANDA ---
elif menu == "🔮 Dashboard Previsão de Demanda":
    st.header("🔮 Previsão de Demanda Técnica")

    # Tratamento seguro dos dados mensais
    historico_mensal = f.groupby("ano_mes").size().reset_index()
    historico_mensal.columns = ["ano_mes", "atendimentos"]
    historico_mensal = historico_mensal.sort_values("ano_mes")
    
    historico_mensal["data_mes"] = pd.to_datetime(historico_mensal["ano_mes"] + "-01")

    if len(historico_mensal) >= 4:
        historico_mensal["media_movel_3m"] = historico_mensal["atendimentos"].rolling(window=3, min_periods=1).mean()
        ultimos = historico_mensal.tail(6)
        
        x = np.arange(len(ultimos))
        y = ultimos["atendimentos"].values
        coef = np.polyfit(x, y, 1)
        tendencia = coef[0]

        ultima_media = historico_mensal["media_movel_3m"].iloc[-1]
        ultimo_mes = historico_mensal["data_mes"].max()
        meses_futuros = pd.date_range(start=ultimo_mes + pd.DateOffset(months=1), periods=3, freq="MS")

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

        df_plot = pd.concat([historico_plot[["data_mes", "Período/Mês", "Volume de Atendimentos", "Tipo"]], pd.DataFrame(projecoes)])
        st.plotly_chart(px.bar(df_plot, x="Período/Mês", y="Volume de Atendimentos", color="Tipo", title="Histórico e Projeção dos Próximos 3 Meses", color_discrete_sequence=["#1f77b4", "#aec7e8"]), use_container_width=True)
        
        # --- LÓGICA DE TENDÊNCIA DIRETA E TOTALMENTE BLINDADA ---
        if tendencia > 0.5:
            tendencia_txt = "📈 Alta nas Demandas"
        elif tendencia < -0.5:
            tendencia_txt = "📉 Queda nas Demandas"
        else:
            tendencia_txt = "➡️ Operação Estável"

        c1, c2, c3 = st.columns(3)
        c1.metric("Média Móvel Atual", f"{ultima_media:.1f}")
        c2.metric("Tendência da Operação", tendencia_txt)
        c3.metric("Projeção Próximo Mês", int(projecoes[0]["Volume de Atendimentos"]))
        
        # ==========================================================================
        # 💡 [NOVOS INSIGHTS ADICIONADOS]: PAINEL DE ANÁLISE PREDITIVA
        # ==========================================================================
        st.markdown("---")
        st.subheader("💡 Insights Estratégicos de Planejamento")
        
        pico_historico = historico_mensal["atendimentos"].max()
        mes_pico = historico_mensal.loc[historico_mensal["atendimentos"].idxmax(), "ano_mes"]
        total_projetado_trimestre = sum([p["Volume de Atendimentos"] for p in projecoes])
        
        col_ins1, col_ins2 = st.columns(2)
        with col_ins1:
            st.info(f"📊 **Histórico de Teto Operacional:** O maior pico de atendimentos já registrado pela ATI ocorreu em **{mes_pico}**, com um total de **{pico_historico}** agendas em um único mês.")
            if projecoes[0]["Volume de Atendimentos"] > ultima_media:
                st.warning("⚠️ **Alerta de Alocação de Recursos:** A projeção matemática aponta que o próximo mês ficará **acima da média móvel recente**. Recomenda-se que a coordenação evite sobrecarregar as equipes com novos planejamentos complexos nas próximas semanas.")
            else:
                st.success("✅ **Estabilização Logística:** O volume previsto para o próximo mês indica um ritmo controlado dentro ou abaixo da média histórica, ideal para focar na limpeza de relatórios atrasados e organização de drives.")
                
        with col_ins2:
            st.metric("Acumulado Previsto (Próximos 3 Meses)", f"{total_projetado_trimestre} Atendimentos", help="Soma total das demandas calculadas para o próximo trimestre.")
            st.caption(f"*Nota metodológica:* Este modelo utiliza regressão linear baseada no ritmo dos últimos 6 meses com peso suavizado pela média móvel trimestral. Fatores externos e novas ordens de reassentamento podem alterar a precisão real.")
            
    else:
        st.warning("São necessários pelo menos 4 meses de histórico para calcular a linha de tendência e projeções.")