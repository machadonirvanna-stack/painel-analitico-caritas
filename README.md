# 📊 Sistema de análise dos dados de assessoramento — ATI Cáritas

Este sistema foi desenvolvido para transformar dados brutos de registos de atendimento em conhecimento estratégico, otimizando a tomada de decisão, a gestão de equipas e a defesa de direitos socioambientais da Assessoria Técnica Independente (ATI) da Cáritas.

O painel está publicado e disponível na nuvem através do **Streamlit Community Cloud**.

---

## 🚀 Funcionalidades e Dashboards

O sistema está estruturado em módulos especializados para responder a diferentes níveis de gestão:

- **✨ Insights Automáticos:** Alertas rápidos baseados no princípio de Pareto (concentração de esforço técnico), avisos automáticos de sobrecarga de assessores e notificações de inconsistências operacionais diretamente na capa.
- **📊 Dashboard Executivo:** Visão de alto nível com indicadores mensais comparativos (Month-over-Month), distribuição de público por sexo (gênero) e classificação automatizada da natureza do atendimento (Coletivas, Individuais, Demandas Espontâneas e Parcerias).
- **📋 Qualidade, Evidências e Encaminhamentos:** Auditoria ativa de compliance para monitorizar se as agendas possuem relatórios cadastrados e ficheiros salvos no Drive, além da rastreabilidade e destinação de todas as demandas encaminhadas externamente.
- **⏱️ Dashboard Operacional (Sazonalidade):** Análise volumétrica por mês do ano, identificação dos dias mais críticos da semana e cruzamento analítico de formato por complexidade média.
- **👥 Dashboard Equipe & Carga:** Matriz de equilíbrio entre horas dedicadas e complexidade das ações, ranking de parcerias mais recorrentes (trabalho em duplas) e matriz de calendário semanal de alocação por técnico.
- **🗺️ Dashboard Territorial:** Mapeamento geográfico analítico das comunidades acompanhadas e ranking de territórios com maior aceleração ou crescimento de demandas recentes.
- **⚠️ Dashboard Vulnerabilidades:** Ordenação de priorização territorial baseada no **Score Único de Risco Combinado** (Vulnerabilidade + Reincidência + Casos Sensíveis).
- **🔮 Previsão de Demanda Técnica:** Modelo preditivo estatístico suavizado por média móvel trimestral que projeta a tendência da operação para os próximos 3 meses.

---

## 🛠️ Como Executar o Projeto Localmente

Se pretender rodar o painel no seu computador pessoal, siga os passos abaixo:

### 1. Pré-requisitos
Certifique-se de que tem o Python instalado na sua máquina. A estrutura de ficheiros na pasta do seu projeto deve ser a seguinte:

```text
├── app.py                       # Código principal do Streamlit
├── requirements.txt             # Arquivo de dependências
└── 02. Registro de Assessoramento - Atualizado  (respostas).xlsx  # Planilha de dados original
