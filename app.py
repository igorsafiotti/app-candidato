import streamlit as st
import fitz  # PyMuPDF
import json
import google.generativeai as genai
from typing import Dict, Any, List, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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

# Campo para API Key do Gemini (apenas para Premium)
api_key = st.sidebar.text_input(
    "🔑 API Key do Google Gemini (Opcional - Premium)",
    type="password",
    help="Insira sua API Key do Google AI Studio para Análise Profunda",
    placeholder="Cole sua API Key aqui..."
)

st.sidebar.divider()

# Botões de análise
analyze_button_free = st.sidebar.button("⚡ Análise Rápida (Grátis)")
analyze_button_premium = st.sidebar.button("💎 Análise Profunda (Premium)", type="primary")

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

def analyze_resume_free(resume_text: str, job_desc: str, pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza análise estatística local do currículo usando TF-IDF.
    Análise rápida e gratuita sem IA.
    """
    # Stopwords básicas em português e inglês
    stopwords = {
        "a", "o", "e", "de", "do", "da", "em", "para", "com", "um", "uma", "é", "que",
        "por", "como", "ou", "se", "não", "na", "no", "as", "os", "mais", "ser", "ter",
        "estar", "ao", "dos", "das", "à", "pelo", "pela", "sua", "seu", "são", "nos",
        "nas", "mas", "também", "muito", "já", "ainda", "bem", "sobre", "entre", "todos",
        "essa", "esse", "isso", "aquilo", "quem", "qual", "quais", "cujo", "cujos",
        "neste", "nesta", "nesse", "nessa", "pelos", "pelas", "das", "dos", "me", "te",
        "lhe", "nos", "vos", "lhes", "minha", "teu", "nossa", "vossa", "meus", "teus",
        "nossos", "vossos", "minhas", "tuas", "nossas", "vossas", "qualquer", "alguns",
        "alguma", "algum", "algumas", "pouco", "pouca", "poucos", "poucas", "grande",
        "grandes", "pequeno", "pequena", "pequenos", "pequenas", "outro", "outra",
        "outros", "outras", "todo", "toda", "todos", "todas", "mesmo", "mesma", "mesmos",
        "mesmas", "próprio", "própria", "próprios", "próprias", "pode", "podem", "puder",
        "pudesse", "podia", "podiam", "poder", "poderá", "poderão", "deve", "devem",
        "dever", "deverá", "deverão", "devia", "deviam", "the", "and", "for", "are",
        "but", "not", "you", "all", "can", "had", "her", "was", "one", "our", "out",
        "with", "this", "that", "from", "they", "will", "have", "been", "more", "when"
    }
    
    # Vetorização TF-IDF
    try:
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        similarity = 0.0
    
    # Converter similaridade (0-1) para score (0-10)
    score_prontidao = similarity * 10
    
    # Extração básica de palavras-chave (palavras com mais de 4 letras)
    job_words = re.findall(r'\b[a-zA-Zá-úÁ-Úà-ãÂ-ÎÔÛâ-îôûä-ïöüçÇ]{5,}\b', job_desc.lower())
    job_words = [word for word in job_words if word not in stopwords]
    job_words = list(set(job_words))  # Remover duplicatas
    
    resume_lower = resume_text.lower()
    hard_skills_encontradas = [word for word in job_words if word in resume_lower]
    hard_skills_ausentes = [word for word in job_words if word not in resume_lower]
    
    # Veredito baseado no score
    if score_prontidao >= 7:
        veredito = "Currículo com boa compatibilidade estatística. Análise semântica detalhada disponível na versão Premium."
    elif score_prontidao >= 4:
        veredito = "Currículo com compatibilidade moderada. Recomenda-se análise profunda para avaliação contextual."
    else:
        veredito = "Currículo com baixa compatibilidade estatística. Considere revisar palavras-chave e estrutura."
    
    return {
        "score_prontidao": score_prontidao,
        "veredito_10_segundos": veredito,
        "diagnostico_ats": {
            "risco_parsing": pdf_metadata.get('risco_parsing', 'Médio'),
            "falhas_estruturais": pdf_metadata.get('falhas_estruturais', [])
        },
        "analise_palavras_chave": {
            "hard_skills_vaga": job_words[:20],  # Limitar a 20
            "hard_skills_encontradas": hard_skills_encontradas[:20],
            "hard_skills_ausentes": hard_skills_ausentes[:20],
            "compatibilidade_semantica": "Não avaliada (versão Free)",
            "orientacao_insercao": f"Adicione os {len(hard_skills_ausentes)} termos ausentes para melhorar compatibilidade."
        },
        "orientacao_resultados_xyz": {
            "percentual_itens_com_metricas": None,
            "adjetivos_esponjosos_detectados": None,
            "frase_original_curriculo": None,
            "amostra_reescrita_xyz": None,
            "is_premium": False
        },
        "is_premium": False
    }

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
- Seja extremamente rigoroso na correspondência de skills. Se a vaga pede 'Gestão de Projetos' no contexto de Engenharia Civil, NÃO considere correspondência se o currículo possui 'Gestão de Projetos Digitais' ou metodologias ágeis de software. Avalie a semântica, não apenas a palavra isolada.

2. ANÁLISE SEMÂNTICA DE CONTEXTO
- Avalie o contexto real da área de atuação. Ex: "Gestão de Projetos Digitais" é DIFERENTE de "Gestão de Obras Civis" - não dê pontuação alta apenas por coincidência de palavras.
- Verifique se as competências do candidato são semanticamente compatíveis com a área específica da vaga.
- Um gestor de TI aplicando para Engenharia Civil NÃO deve ter nota alta apenas por ter a palavra "projetos" no currículo.

3. FILTRAGEM DE ADJETIVOS ESPONJOSOS
- Detecte adjetivos de personalidade subjetiva não comprovável: "proativo", "dinâmico", "dedicado", "perfeccionista", "visão sistêmica", "excelente comunicação", "criativo", "inovador", "autodidata", "flexível", "apaixonado", "entusiasmado", "motivado", "comprometido", "resiliente", "detalhista", "organizado", "comunicativo", "líder", "trabalho em equipe", "focado", "orientado a resultados", "pensamento crítico".
- Adjetivos sem evidência factual têm peso ZERO.
- NÃO penalize termos de escopo de negócios (como "parcerias estratégicas" ou "gestão estratégica"), focando apenas em adjetivos de traços de personalidade (ex: "proativo", "perfeccionista").

4. METODOLOGIA 'GOOGLE XYZ' E ORIENTAÇÃO A RESULTADOS
- Identifique UMA frase REAL do currículo do usuário que contenha adjetivos esponjosos ou falta de métricas.
- Reescreva essa mesma frase usando a fórmula Google XYZ: "Conquistei [X], medido por [Y], realizando [Z]".
- A reescrita DEVE ser baseada em uma frase REAL extraída do currículo, NÃO texto inventado.
- PROIBIDO INVENTAR OU ALUCINAR NÚMEROS. Se a frase original não possuir métricas reais, reescreva a estrutura XYZ utilizando placeholders claros como [INSERIR %], [INSERIR VALOR R$] ou [INSERIR QUANTIDADE], forçando o usuário a preencher com seus dados reais.

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
            
            result = json.loads(response.text)
            result["is_premium"] = True  # Marcar como análise premium
            return result
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
    Adapta a exibição baseado em se é análise Free ou Premium.
    """
    is_premium = analysis.get('is_premium', False)
    
    if is_premium:
        st.header("📊 Resultados da Análise Profunda (Premium)")
    else:
        st.header("📊 Resultados da Análise Rápida (Grátis)")
    
    # Score de Prontidão
    col1, col2, col3 = st.columns(3)
    with col1:
        score_valor = min(analysis.get('score_prontidao', 0), 10.0)  # Garantir máximo de 10
        delta_text = "Análise semântica com IA" if is_premium else "Análise estatística TF-IDF"
        st.metric(
            label="Score de Prontidão",
            value=f"{score_valor:.1f}/10",
            delta=delta_text
        )
    
    # Veredito de 10 segundos
    veredito = analysis.get('veredito_10_segundos', 'Não disponível')
    score_normalizado = min(analysis.get('score_prontidao', 0), 10.0) / 10.0  # Normalizar para 0-1
    if score_normalizado >= 0.8:
        st.success(f"**Veredito (10 segundos):** {veredito}")
    elif score_normalizado >= 0.5:
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
    st.subheader("🔑 Análise de Hard Skills" + (" (Semântica)" if is_premium else " (Estatística)"))
    palavras_chave = analysis.get('analise_palavras_chave', {})
    
    hard_skills_vaga = palavras_chave.get('hard_skills_vaga', [])
    hard_skills_encontradas = palavras_chave.get('hard_skills_encontradas', [])
    hard_skills_ausentes = palavras_chave.get('hard_skills_ausentes', [])
    compatibilidade = palavras_chave.get('compatibilidade_semantica', 'Não avaliada')
    
    if is_premium:
        if compatibilidade == "Alta":
            st.success(f"**Compatibilidade Semântica:** {compatibilidade}")
        elif compatibilidade == "Média":
            st.warning(f"**Compatibilidade Semântica:** {compatibilidade}")
        else:
            st.error(f"**Compatibilidade Semântica:** {compatibilidade}")
    else:
        st.info("⚡ **Versão Grátis:** Análise estatística baseada em TF-IDF. Para avaliação semântica contextual, use a Análise Profunda (Premium).")
    
    st.metric(
        label="Hard Skills da Vaga",
        value=f"{len(hard_skills_encontradas)}/{len(hard_skills_vaga)}",
        delta=f"{(len(hard_skills_encontradas)/len(hard_skills_vaga)*100) if hard_skills_vaga else 0:.0f}% encontradas"
    )
    
    # Exibir Hard Skills encontradas
    if hard_skills_encontradas:
        st.write(f"**{len(hard_skills_encontradas)} Hard Skills Encontradas:**")
        for skill in hard_skills_encontradas:
            st.success(f"✓ {skill}")
    else:
        st.warning("⚠ Nenhuma Hard Skill da vaga encontrada no currículo")
    
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
    if is_premium:
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
    else:
        # Banner de Upsell para versão Free
        st.subheader("📈 Orientação de Resultados (Google XYZ)")
        st.info("🔒 **Recurso Premium:** Identificação de adjetivos esponjosos e reescrita de impacto Google XYZ estão disponíveis apenas na Análise Profunda.")
        
        st.markdown("""
        ### 💎 Recursos Exclusivos Premium:
        - ✅ Detecção avançada de adjetivos esponjosos com IA
        - ✅ Reescrita Google XYZ baseada em frases reais do currículo
        - ✅ Análise semântica contextual de área de atuação
        - ✅ Extração inteligente de Hard Skills reais
        - ✅ Compatibilidade semântica (não apenas estatística)
        
        Clique em **"💎 Análise Profunda (Premium)"** para acessar todos os recursos.
        """)

# Lógica principal
if analyze_button_free or analyze_button_premium:
    if not uploaded_file:
        st.sidebar.error("Por favor, faça upload de um currículo em PDF.")
    elif not job_description:
        st.sidebar.error("Por favor, insira a descrição da vaga.")
    elif analyze_button_premium and not api_key:
        st.sidebar.error("Por favor, insira sua API Key do Google Gemini para Análise Profunda.")
    else:
        # Extrair texto do PDF (análise local)
        with st.spinner("Extraindo texto do PDF e avaliando estrutura ATS..."):
            resume_text, pdf_metadata = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            if analyze_button_free:
                # Análise Free (TF-IDF)
                with st.spinner("Realizando Análise Rápida (TF-IDF)..."):
                    analysis = analyze_resume_free(resume_text, job_description, pdf_metadata)
                
                if analysis:
                    display_results(analysis)
            
            elif analyze_button_premium:
                # Análise Premium (Gemini)
                with st.spinner("Realizando Análise Profunda com Gemini AI..."):
                    analysis = analyze_resume_with_gemini(resume_text, job_description, api_key, pdf_metadata)
                
                if analysis:
                    display_results(analysis)
else:
    # Tela inicial
    st.title("🚀 ResumePilot")
    st.markdown("""
    ### Bem-vindo ao ResumePilot - Modelo Freemium!
    
    Este é um sistema de análise de currículos com dois níveis de análise:
    
    ---
    
    ### ⚡ Análise Rápida (Grátis)
    - ✅ **Análise ATS Local (PyMuPDF)**: Avaliação estrutural do PDF
    - ✅ **Análise Estatística (TF-IDF)**: Similaridade de texto básica
    - ✅ **Extração de Palavras-chave**: Identificação de termos relevantes
    - ✅ **Score de Prontidão**: Nota baseada em compatibilidade estatística
    - ✅ **100% Local**: Rápido, sem API Key, sem custos
    
    ---
    
    ### 💎 Análise Profunda (Premium)
    - ✅ **Tudo da versão Grátis** + recursos avançados com IA
    - ✅ **Análise Semântica (Gemini AI)**: Compreensão contextual de Hard Skills
    - ✅ **Compatibilidade Real**: Avalia contexto de área (ex: TI vs Engenharia Civil)
    - ✅ **Detecção de Adjetivos Esponjosos**: Identifica clichês sem evidência
    - ✅ **Metodologia Google XYZ**: Reescrita de frases reais com métricas
    - ✅ **Score Contextual**: Nota baseada em semântica, não apenas palavras
    
    ---
    
    **Como usar:**
    1. Faça upload do seu currículo em PDF na barra lateral
    2. Cole a descrição da vaga que você deseja aplicar
    3. Escolha **"⚡ Análise Rápida (Grátis)"** para análise rápida local
    4. Ou insira sua API Key e clique em **"💎 Análise Profunda (Premium)"** para análise completa
    
    📌 **Nota:** Você pode obter sua API Key gratuita em [Google AI Studio](https://makersuite.google.com/app/apikey)
    
    🔒 **Privacidade:** A análise estrutural é sempre feita localmente. Apenas a versão Premium envia texto ao Gemini para análise semântica.
    """)
