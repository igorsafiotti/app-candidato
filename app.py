import streamlit as st
import fitz  # PyMuPDF
import json
import google.generativeai as genai
from typing import Dict, Any, List, Tuple
import re

# Configuração da página
st.set_page_config(
    page_title="ResumePilot",
    page_icon="📄",
    layout="wide"
)

# Sidebar para upload e inputs
st.sidebar.title("ResumePilot")
st.sidebar.markdown("Análise Híbrida de Currículos")

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

def extract_text_from_pdf(pdf_file) -> Tuple[str, Dict[str, Any]]:
    """
    Extrai todo o texto cru de um arquivo PDF usando PyMuPDF (fitz).
    Retorna o texto e um dicionário com metadados de parsing.
    """
    try:
        pdf_document = fitz.open(stream=pdf_file.read(), filetype="pdf")
        text = ""
        total_pages = len(pdf_document)
        pages_with_text = 0
        total_blocks = 0
        
        for page_num in range(total_pages):
            page = pdf_document[page_num]
            page_text = page.get_text()
            text += page_text
            
            # Contar blocos de texto para avaliar complexidade
            blocks = page.get_text("blocks")
            if blocks:
                total_blocks += len(blocks)
            
            if page_text.strip():
                pages_with_text += 1
        
        pdf_document.close()
        
        # Avaliar risco de parsing
        parsing_risk = "Baixo"
        structural_issues = []
        
        if pages_with_text < total_pages:
            parsing_risk = "Médio"
            structural_issues.append(f"{total_pages - pages_with_text} página(s) sem texto detectado")
        
        if total_blocks > total_pages * 10:
            parsing_risk = "Alto"
            structural_issues.append("Alta complexidade de blocos de texto (possível uso de múltiplas colunas)")
        
        return text, {
            "risco_parsing": parsing_risk,
            "falhas_estruturais": structural_issues,
            "paginas_com_texto": pages_with_text,
            "total_paginas": total_pages
        }
    except Exception as e:
        st.error(f"Erro ao extrair texto do PDF: {str(e)}")
        return "", {"risco_parsing": "Alto", "falhas_estruturais": [str(e)], "paginas_com_texto": 0, "total_paginas": 0}

def analyze_resume_with_gemini(resume_text: str, job_desc: str, api_key: str, pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza análise semântica do currículo usando Google Gemini.
    Extrai apenas Hard Skills reais e entende contexto semântico.
    """
    genai.configure(api_key=api_key)
    
    system_instruction = """Você é o motor de inteligência analítica do ResumePilot, atuando como um Auditor Técnico de ATS e Recrutador Corporativo Sênior. Sua função é analisar o currículo em PDF/texto fornecido pelo usuário e compará-lo rigorosamente com a Descrição da Vaga (Job Description) enviada.

1. EXTRAÇÃO DE HARD SKILLS (APENAS TECNOLOGIAS E METODOLOGIAS)
- Extraia APENAS Hard Skills reais da vaga: tecnologias, ferramentas, frameworks, linguagens de programação, metodologias específicas, certificações técnicas.
- IGNORE completamente: verbos, pronomes, adjetivos genéricos, termos vagos como "gestão", "projetos", "comunicação" isoladamente.
- Exemplo de Hard Skills válidas: Python, React, Scrum, AWS, Docker, Jenkins, Agile, Jira, SQL, etc.
- Exemplo de termos a IGNORAR: "gestão de projetos", "comunicação eficaz", "trabalho em equipe", "liderança", "organização".

2. ANÁLISE SEMÂNTICA DE CONTEXTO
- Avalie o contexto real da área de atuação. Ex: "Gestão de Projetos Digitais" é DIFERENTE de "Gestão de Obras Civis" - não dê pontuação alta apenas por coincidência de palavras.
- Verifique se as competências do candidato são semanticamente compatíveis com a área específica da vaga.
- Um gestor de TI aplicando para Engenharia Civil NÃO deve ter nota alta apenas por ter a palavra "projetos" no currículo.

3. FILTRAGEM DE ADJETIVOS ESPONJOSOS
- Detecte adjetivos de personalidade subjetiva não comprovável: "proativo", "dinâmico", "dedicado", "perfeccionista", "visão sistêmica", "excelente comunicação", "criativo", "inovador", "autodidata", "flexível", "apaixonado", "entusiasmado", "motivado", "comprometido", "resiliente", "detalhista", "organizado", "comunicativo", "líder", "trabalho em equipe", "focado", "orientado a resultados", "pensamento crítico".
- Adjetivos sem evidência factual têm peso ZERO.

4. METODOLOGIA 'GOOGLE XYZ' E ORIENTAÇÃO A RESULTADOS
- Identifique UMA frase REAL do currículo do usuário que contenha adjetivos esponjosos ou falta de métricas.
- Reescreva essa mesma frase usando a fórmula Google XYZ: "Conquistei [X], medido por [Y], realizando [Z]".
- A reescrita DEVE ser baseada em uma frase REAL extraída do currículo, NÃO texto inventado.

5. MÉTRICAS E EVIDÊNCIAS QUANTITATIVAS
- Detecte presença de métricas numéricas: porcentagens (%), valores monetários (R$, $), números grandes, escala (mil, milhões).
- Verifique verbos de ação no passado e presença explícita de métricas.

6. SCORE DE PRONTIDÃO (0 a 10)
- Calcule uma nota baseada em:
  * Proporção de Hard Skills da vaga presentes no currículo (peso 40%)
  * Compatibilidade semântica de contexto/área de atuação (peso 30%)
  * Presença de métricas e evidências quantitativas (peso 20%)
  * Penalização por adjetivos esponjosos (peso 10%)
- O score DEVE refletir o contexto semântico real, não apenas coincidência de palavras.

7. FORMATO DA RESPOSTA (JSON)
Retorne a análise estritamente no formato JSON abaixo, sem marcações markdown como ```json, apenas o objeto:
{
  "score_prontidao": 0.0,
  "veredito_10_segundos": "Resumo de impacto executivo considerando contexto semântico",
  "diagnostico_ats": {
    "risco_parsing": "Baixo/Médio/Alto",
    "falhas_estruturais": ["Lista de falhas"]
  },
  "analise_palavras_chave": {
    "hard_skills_vaga": ["Lista de Hard Skills reais da vaga"],
    "hard_skills_encontradas": ["Hard Skills encontradas no currículo"],
    "hard_skills_ausentes": ["Hard Skills ausentes no currículo"],
    "compatibilidade_semantica": "Alta/Média/Baixa - análise de contexto de área",
    "orientacao_insercao": "Instrução contextualizada"
  },
  "orientacao_resultados_xyz": {
    "percentual_itens_com_metricas": 0,
    "adjetivos_esponjosos_detectados": ["Adjetivos vazios"],
    "frase_original_curriculo": "Frase REAL extraída do currículo",
    "amostra_reescrita_xyz": {
      "antes": "Frase original do currículo",
      "depois": "Frase XYZ reescrita baseada na frase original"
    }
  }
}"""
    
    prompt = f"""
Currículo do Candidato:
{resume_text}

Descrição da Vaga:
{job_desc}

Diagnóstico ATS Local (PyMuPDF):
- Risco de Parsing: {pdf_metadata.get('risco_parsing', 'Médio')}
- Falhas Estruturais: {pdf_metadata.get('falhas_estruturais', [])}

Analise o currículo conforme as diretrizes acima, focando em:
1. Extração APENAS de Hard Skills reais (tecnologias, metodologias)
2. Análise semântica de contexto (área de atuação)
3. Detecção de adjetivos esponjosos
4. Extração de uma frase REAL do currículo para reescrita XYZ
5. Cálculo de score considerando contexto semântico real

Retorne o JSON.
"""
    
    # Lista de modelos para tentar em ordem de preferência
    models_to_try = ["gemini-2.5-flash", "gemini-1.5-flash", "gemini-flash-latest"]
    
    for model_name in models_to_try:
        try:
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_instruction
            )
            
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    response_mime_type="application/json"
                )
            )
            
            return json.loads(response.text)
        except Exception as e:
            if model_name == models_to_try[-1]:
                # Última tentativa falhou
                st.error(f"Erro ao chamar a API Gemini: {str(e)}")
                return None
            # Tentar próximo modelo
            continue
    
    return None

def display_results(analysis: Dict[str, Any]):
    """
    Exibe os resultados da análise no dashboard do Streamlit.
    """
    st.header("📊 Resultados da Análise")
    
    # Score de Prontidão
    col1, col2, col3 = st.columns(3)
    with col1:
        score_valor = analysis.get('score_prontidao', 0) * 10
        st.metric(
            label="Score de Prontidão",
            value=f"{score_valor:.1f}/10",
            delta="Baseado em análise semântica"
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
    st.subheader("🔍 Diagnóstico ATS (PyMuPDF)")
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
    
    # Análise de Palavras-Chave (Hard Skills)
    st.subheader("🔑 Análise de Hard Skills (Semântica)")
    palavras_chave = analysis.get('analise_palavras_chave', {})
    
    hard_skills_vaga = palavras_chave.get('hard_skills_vaga', [])
    hard_skills_encontradas = palavras_chave.get('hard_skills_encontradas', [])
    hard_skills_ausentes = palavras_chave.get('hard_skills_ausentes', [])
    compatibilidade = palavras_chave.get('compatibilidade_semantica', 'Não avaliada')
    
    if compatibilidade == "Alta":
        st.success(f"**Compatibilidade Semântica:** {compatibilidade}")
    elif compatibilidade == "Média":
        st.warning(f"**Compatibilidade Semântica:** {compatibilidade}")
    else:
        st.error(f"**Compatibilidade Semântica:** {compatibilidade}")
    
    st.metric(
        label="Hard Skills da Vaga",
        value=f"{len(hard_skills_encontradas)}/{len(hard_skills_vaga)}",
        delta=f"{(len(hard_skills_encontradas)/len(hard_skills_vaga)*100) if hard_skills_vaga else 0:.0f}% encontradas"
    )
    
    if hard_skills_ausentes:
        st.write(f"**{len(hard_skills_ausentes)} Hard Skills Ausentes:**")
        for skill in hard_skills_ausentes:
            st.warning(f"• {skill}")
    else:
        st.success("✅ Todas as Hard Skills da vaga estão presentes")
    
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
        st.write(f"**{len(adjetivos)} Adjetivos Esponjosos Detectados (sem evidência factual):**")
        for adjetivo in adjetivos:
            st.warning(f"• {adjetivo}")
    else:
        st.success("✅ Nenhum adjetivo esponjoso detectado")
    
    # Frase original do currículo
    frase_original = xyz.get('frase_original_curriculo', '')
    if frase_original:
        st.info(f"**Frase Original do Currículo:** {frase_original}")
    
    # Amostra de Reescrita XYZ
    amostra = xyz.get('amostra_reescrita_xyz', {})
    if amostra:
        st.subheader("✨ Amostra de Reescrita XYZ (Baseada em Frase Real)")
        col_antes, col_depois = st.columns(2)
        
        with col_antes:
            st.markdown("**Antes (Frase Original):**")
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
        # Extrair texto do PDF (análise local)
        with st.spinner("Extraindo texto do PDF e avaliando estrutura ATS..."):
            resume_text, pdf_metadata = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            # Análise semântica com Gemini
            with st.spinner("Analisando currículo com Gemini AI (análise semântica)..."):
                analysis = analyze_resume_with_gemini(resume_text, job_description, api_key, pdf_metadata)
            
            if analysis:
                # Exibir resultados
                display_results(analysis)
else:
    # Tela inicial
    st.title("🚀 ResumePilot")
    st.markdown("""
    ### Bem-vindo ao ResumePilot!
    
    Este é um sistema híbrido de análise de currículos que combina:
    
    - ✅ **Análise ATS Local (PyMuPDF)**: Avaliação estrutural do PDF sem IA
    - ✅ **Análise Semântica (Gemini AI)**: Compreensão contextual de Hard Skills e área de atuação
    - ✅ **Detecção de Adjetivos Esponjosos**: Identificação de clichês sem evidência factual
    - ✅ **Metodologia Google XYZ**: Reescrita de frases reais do currículo com métricas
    - ✅ **Score Contextual**: Nota baseada em compatibilidade semântica real, não apenas palavras-chave
    
    **Como usar:**
    1. Faça upload do seu currículo em PDF na barra lateral
    2. Cole a descrição da vaga que você deseja aplicar
    3. Insira sua API Key do Google Gemini
    4. Clique em "Analisar Currículo"
    
    📌 **Nota:** Você pode obter sua API Key gratuita em [Google AI Studio](https://makersuite.google.com/app/apikey)
    
    🔒 **Privacidade:** A análise estrutural é feita localmente. Apenas o texto é enviado ao Gemini para análise semântica.
    """)
