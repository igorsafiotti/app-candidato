import streamlit as st
import fitz  # PyMuPDF
import json
import google.generativeai as genai
from typing import Dict, Any

# Configuração da página
st.set_page_config(
    page_title="ResumePilot AI",
    page_icon="📄",
    layout="wide"
)

# Sidebar para upload e inputs
st.sidebar.title("ResumePilot AI")
st.sidebar.markdown("Análise Inteligente de Currículos")

# Upload do PDF
uploaded_file = st.sidebar.file_uploader(
    "Upload do Currículo (PDF)",
    type=['pdf'],
    help="Carregue seu currículo em formato PDF"
)

# Text area para a descrição da vaga
job_description = st.sidebar.text_area(
    "Descrição da Vaga (Job Description)",
    height=300,
    placeholder="Cole aqui o texto completo da descrição da vaga..."
)

# Campo para API Key do Gemini
api_key = st.sidebar.text_input(
    "🔑 API Key do Google Gemini",
    type="password",
    help="Insira sua API Key do Google AI Studio",
    placeholder="Cole sua API Key aqui..."
)

# Botão de análise
analyze_button = st.sidebar.button("Analisar Currículo", type="primary")

def extract_text_from_pdf(pdf_file) -> str:
    """
    Extrai todo o texto cru de um arquivo PDF usando PyMuPDF (fitz).
    """
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text += page.get_text()
        pdf_document.close()
        return text
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {str(e)}")
        return ""

def analyze_resume_with_gemini(resume_text: str, job_desc: str, api_key: str) -> Dict[str, Any]:
    """
    Envia o currículo e a descrição da vaga para o Google Gemini e retorna a análise.
    """
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-pro",
            system_instruction="""Você é o motor de inteligência analítica do ResumePilot AI, atuando como um Auditor Técnico de ATS e Recrutador Corporativo Sênior. Sua função é analisar o currículo em PDF/texto fornecido pelo usuário e compará-lo rigorosamente com a Descrição da Vaga (Job Description) enviada.
1. DIRETRIZES TÉCNICAS DE PARSING E ESTRUTURA ATS
- Estrutura e Diagramação: Penalize o uso de múltiplas colunas, caixas de texto flutuantes, tabelas sem borda e elementos gráficos decorativos.
- Cabeçalhos de Seção: Exija nomenclaturas canônicas (ex: "Histórico Profissional", "Formação Acadêmica").
- Datas: Verifique se as datas seguem padrões numéricos ou textuais homologados.
2. FILTRAGEM DE ADJETIVOS ESPONJOSOS
- Ignore e expurgue qualquer adjetivo de personalidade ou autoavaliação subjetiva não comprovável (ex: "proativo", "criativo"). Adjetivos sem evidência factual têm peso ZERO.
3. METODOLOGIA 'GOOGLE XYZ' E ORIENTAÇÃO A RESULTADOS
- Avalie as experiências exigindo a fórmula Google XYZ: "Conquistei [X], medido por [Y], realizando [Z]".
- Verifique verbos de ação no passado e presença explícita de métricas numéricas.
4. GESTÃO DE PALAVRAS-CHAVE (SEM KEYWORD STUFFING)
- Mapeie a ausência de Hard Skills e certificações presentes na Vaga.
- PROIBIDO INCENTIVAR KEYWORD STUFFING. Recomende a inserção contextual das palavras diretamente nas frases de experiência, com verbos de ação.
5. FORMATO DA RESPOSTA (JSON)
Retorne a análise estritamente no formato JSON abaixo, sem marcações markdown como ```json, apenas o objeto: { "score_prontidao": 0.0, "veredito_10_segundos": "Resumo de impacto executivo", "diagnostico_ats": { "risco_parsing": "Baixo/Médio/Alto", "falhas_estruturais": ["Lista de falhas"] }, "analise_palavras_chave": { "termos_ausentes_criticos": ["Termos faltantes"], "orientacao_insercao": "Instrução contextualizada" }, "orientacao_resultados_xyz": { "percentual_itens_com_metricas": 0, "adjetivos_esponjosos_detectados": ["Adjetivos vazios"], "amostra_reescrita_xyz": { "antes": "Frase fraca", "depois": "Frase XYZ reescrita" } } }"""
        )
        
        prompt = f"""
Currículo do Candidato:
{resume_text}

Descrição da Vaga:
{job_desc}

Analise o currículo conforme as diretrizes e retorne o JSON.
"""
        
        response = model.generate_content(prompt)
        return json.loads(response.text)
    except Exception as e:
        st.error(f"Erro ao chamar a API Gemini: {str(e)}")
        return None

def display_results(analysis: Dict[str, Any]):
    """
    Exibe os resultados da análise no dashboard do Streamlit.
    """
    st.header("📊 Resultados da Análise")
    
    # Score de Prontidão
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            label="Score de Prontidão",
            value=f"{analysis.get('score_prontidao', 0) * 100:.1f}%",
            delta="Baseado na análise ATS"
        )
    
    # Veredito de 10 segundos
    veredito = analysis.get('veredito_10_segundos', 'Não disponível')
    if analysis.get('score_prontidao', 0) >= 0.8:
        st.success(f"**Veredito (10 segundos):** {veredito}")
    elif analysis.get('score_prontidao', 0) >= 0.5:
        st.warning(f"**Veredito (10 segundos):** {veredito}")
    else:
        st.error(f"**Veredito (10 segundos):** {veredito}")
    
    st.divider()
    
    # Diagnóstico ATS
    st.subheader("🔍 Diagnóstico ATS")
    diagnostico = analysis.get('diagnostico_ats', {})
    risco_parsing = diagnostico.get('risco_parsing', 'Não disponível')
    
    if risco_parsing == "Baixo":
        st.success(f"Risco de Parsing: {risco_parsing}")
    elif risco_parsing == "Médio":
        st.warning(f"Risco de Parsing: {risco_parsing}")
    else:
        st.error(f"Risco de Parsing: {risco_parsing}")
    
    falhas = diagnostico.get('falhas_estruturais', [])
    if falhas:
        st.write("**Falhas Estruturais Detectadas:**")
        for falha in falhas:
            st.error(f"• {falha}")
    else:
        st.success("✅ Nenhuma falha estrutural crítica detectada")
    
    st.divider()
    
    # Análise de Palavras-Chave
    st.subheader("🔑 Análise de Palavras-Chave")
    palavras_chave = analysis.get('analise_palavras_chave', {})
    
    termos_ausentes = palavras_chave.get('termos_ausentes_criticos', [])
    if termos_ausentes:
        st.write("**Termos Ausentes Críticos:**")
        for termo in termos_ausentes:
            st.warning(f"• {termo}")
    else:
        st.success("✅ Todos os termos críticos da vaga estão presentes")
    
    orientacao = palavras_chave.get('orientacao_insercao', '')
    if orientacao:
        st.info(f"**Orientação para Inserção:** {orientacao}")
    
    st.divider()
    
    # Orientação Resultados XYZ
    st.subheader("📈 Orientação de Resultados (Google XYZ)")
    xyz = analysis.get('orientacao_resultados_xyz', {})
    
    percentual_metricas = xyz.get('percentual_itens_com_metricas', 0)
    st.metric(
        label="Itens com Métricas",
        value=f"{percentual_metricas}%",
        delta="Baseado em fórmula XYZ"
    )
    
    adjetivos = xyz.get('adjetivos_esponjosos_detectados', [])
    if adjetivos:
        st.write("**Adjetivos Esponjosos Detectados (sem evidência factual):**")
        for adjetivo in adjetivos:
            st.warning(f"• {adjetivo}")
    else:
        st.success("✅ Nenhum adjetivo esponjoso detectado")
    
    # Amostra de Reescrita XYZ
    amostra = xyz.get('amostra_reescrita_xyz', {})
    if amostra:
        st.subheader("✨ Amostra de Reescrita XYZ")
        col_antes, col_depois = st.columns(2)
        
        with col_antes:
            st.markdown("**Antes (Frase Fraca):**")
            st.error(amostra.get('antes', ''))
        
        with col_depois:
            st.markdown("**Depois (Frase XYZ):**")
            st.success(amostra.get('depois', ''))

# Lógica principal
if analyze_button:
    if not uploaded_file:
        st.sidebar.error("Por favor, faça upload de um currículo em PDF.")
    elif not job_description:
        st.sidebar.error("Por favor, insira a descrição da vaga.")
    elif not api_key:
        st.sidebar.error("Por favor, insira sua API Key do Google Gemini.")
    else:
        # Extrair texto do PDF
        with st.spinner("Extraindo texto do PDF..."):
            resume_text = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            # Enviar para Gemini
            with st.spinner("Analisando currículo com Gemini AI..."):
                analysis = analyze_resume_with_gemini(resume_text, job_description, api_key)
            
            if analysis:
                # Exibir resultados
                display_results(analysis)
else:
    # Tela inicial
    st.title("🚀 ResumePilot AI")
    st.markdown("""
    ### Bem-vindo ao ResumePilot AI!
    
    Este é um sistema inteligente de análise de currículos que utiliza IA para:
    
    - ✅ Analisar a estrutura ATS do seu currículo
    - ✅ Identificar palavras-chave críticas ausentes
    - ✅ Detectar adjetivos esponjosos sem evidência factual
    - ✅ Sugerir reescritas usando a metodologia Google XYZ
    
    **Como usar:**
    1. Faça upload do seu currículo em PDF na barra lateral
    2. Cole a descrição da vaga que você deseja aplicar
    3. Insira sua API Key do Google Gemini
    4. Clique em "Analisar Currículo"
    
    📌 **Nota:** Você pode obter sua API Key gratuita em [Google AI Studio](https://makersuite.google.com/app/apikey)
    """)
