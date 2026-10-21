import streamlit as st
import fitz  # PyMuPDF
import re
from typing import Dict, Any, List, Tuple
import string

# Configuração da página
st.set_page_config(
    page_title="ResumePilot",
    page_icon="📄",
    layout="wide"
)

# Sidebar para upload e inputs
st.sidebar.title("ResumePilot")
st.sidebar.markdown("Análise Local de Currículos")

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

# Botão de análise
analyze_button = st.sidebar.button("Analisar Currículo", type="primary")

# Lista de adjetivos esponjosos em português
ADJETIVOS_ESPONJOSOS = [
    "proativo", "proativa", "proativos", "proativas",
    "dinâmico", "dinâmica", "dinâmicos", "dinâmicas",
    "dedicado", "dedicada", "dedicados", "dedicadas",
    "perfeccionista", "perfeccionistas",
    "visão sistêmica", "visao sistemica",
    "excelente comunicação", "excelente comunicacao",
    "criativo", "criativa", "criativos", "criativas",
    "inovador", "inovadora", "inovadores", "inovadoras",
    "autodidata", "autodidatas",
    "flexível", "flexivel", "flexíveis", "flexiveis",
    "apaixonado", "apaixonada", "apaixonados", "apaixonadas",
    "entusiasmado", "entusiasmada", "entusiasmados", "entusiasmadas",
    "motivado", "motivada", "motivados", "motivadas",
    "comprometido", "comprometida", "comprometidos", "comprometidas",
    "resiliente", "resilientes",
    "detalhista", "detalhistas",
    "organizado", "organizada", "organizados", "organizadas",
    "comunicativo", "comunicativa", "comunicativos", "comunicativas",
    "líder", "lider", "líderes", "lideres",
    "trabalho em equipe", "trabalho em equipa",
    "focado", "focada", "focados", "focadas",
    "orientado a resultados", "orientado a resultado",
    "pensamento crítico", "pensamento critico",
    "solucionador de problemas", "solucionadora de problemas"
]

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

def extract_keywords_from_job_description(job_desc: str) -> List[str]:
    """
    Extrai palavras-chave relevantes da descrição da vaga.
    Filtra stopwords e mantém termos técnicos e substantivos.
    """
    # Stopwords em português (simplificada)
    stopwords = {
        "a", "o", "e", "de", "do", "da", "em", "para", "com", "um", "uma", "é", "que",
        "por", "como", "ou", "se", "não", "na", "no", "as", "os", "mais", "ser", "ter",
        "estar", "ao", "dos", "das", "à", "pelo", "pela", "sua", "seu", "são", "nos",
        "nas", "mas", "também", "muito", "já", "ainda", "bem", "sobre", "entre", "todos",
        "essa", "esse", "essa", "isso", "aquilo", "quem", "qual", "quais", "cujo", "cujos",
        "neste", "nesta", "nesse", "nessa", "pelos", "pelas", "das", "dos", "me", "te",
        "lhe", "nos", "vos", "lhes", "minha", "teu", "nossa", "vossa", "meus", "teus",
        "nossos", "vossos", "minhas", "tuas", "nossas", "vossas", "qualquer", "alguns",
        "alguma", "algum", "algumas", "pouco", "pouca", "poucos", "poucas", "grande",
        "grande", "grandes", "grande", "pequeno", "pequena", "pequenos", "pequenas",
        "outro", "outra", "outros", "outras", "todo", "toda", "todos", "todas", "mesmo",
        "mesma", "mesmos", "mesmas", "próprio", "própria", "próprios", "próprias", "pode",
        "podem", "puder", "pudesse", "podia", "podiam", "poder", "poderá", "poderão",
        "deve", "devem", "dever", "deverá", "deverão", "devia", "deviam", "firme", "forte",
        "sobre", "sob", "trás", "após", "até", "desde", "durante", "mediante", "segundo",
        "conforme", "contrário", "favor", "assim", "logo", "depois", "antes", "agora",
        "hoje", "ontem", "amanhã", "sempre", "nunca", "jamais", "talvez", "quase",
        "apenas", "somente", "tão", "tão", "quanto", "quão", "onde", "quando", "como",
        "porque", "porquê", "por que", "pois", "portanto", "entretanto", "contudo",
        "todavia", "porém", "senão", "caso", "senão", "seja", "sejam", "fosse", "fossem"
    }
    
    # Converter para minúsculas e remover pontuação
    text = job_desc.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Extrair palavras com pelo menos 3 caracteres
    words = re.findall(r'\b[a-zá-úà-ãâêîôûäëïöüç]{3,}\b', text)
    
    # Filtrar stopwords e termos muito comuns
    keywords = [word for word in words if word not in stopwords]
    
    # Remover duplicatas mantendo ordem
    seen = set()
    unique_keywords = []
    for word in keywords:
        if word not in seen:
            seen.add(word)
            unique_keywords.append(word)
    
    return unique_keywords[:50]  # Limitar a 50 palavras-chave mais relevantes

def detect_fluffy_adjectives(resume_text: str) -> List[str]:
    """
    Detecta adjetivos esponjosos no texto do currículo.
    """
    resume_lower = resume_text.lower()
    found_adjectives = []
    
    for adjetivo in ADJETIVOS_ESPONJOSOS:
        if adjetivo in resume_lower:
            found_adjectives.append(adjetivo)
    
    return found_adjectives

def detect_metrics(resume_text: str) -> Tuple[int, List[str]]:
    """
    Detecta métricas numéricas no texto usando regex.
    Retorna o número de ocorrências e exemplos encontrados.
    """
    # Padrões para métricas: porcentagens, valores monetários, números grandes
    patterns = [
        r'\d+%',  # Porcentagens
        r'R\$\s*\d+',  # Valores em Reais
        r'\$\s*\d+',  # Valores em dólar
        r'\d{4,}',  # Números grandes (possíveis anos ou quantidades)
        r'\d+\s*(mil|milhões|milhoes|milhar)',  # Números com escala
        r'\d+\.\d+',  # Números decimais
    ]
    
    metrics_found = []
    total_count = 0
    
    for pattern in patterns:
        matches = re.findall(pattern, resume_text, re.IGNORECASE)
        total_count += len(matches)
        metrics_found.extend(matches[:5])  # Limitar exemplos
    
    return total_count, metrics_found

def calculate_readiness_score(
    keywords_found_ratio: float,
    fluffy_adjectives_count: int,
    metrics_count: int,
    parsing_risk: str
) -> float:
    """
    Calcula o score de prontidão (0 a 10) baseado em vários fatores.
    """
    score = 10.0
    
    # Penalidade por palavras-chave ausentes (até -3 pontos)
    score -= (1.0 - keywords_found_ratio) * 3
    
    # Penalidade por adjetivos esponjosos (até -2 pontos)
    score -= min(fluffy_adjectives_count * 0.3, 2.0)
    
    # Penalidade por falta de métricas (até -2 pontos)
    if metrics_count == 0:
        score -= 2.0
    elif metrics_count < 3:
        score -= 1.0
    
    # Penalidade por risco de parsing alto (até -1 ponto)
    if parsing_risk == "Alto":
        score -= 1.0
    elif parsing_risk == "Médio":
        score -= 0.5
    
    # Garantir que o score fique entre 0 e 10
    return max(0.0, min(10.0, score))

def analyze_resume_locally(resume_text: str, job_desc: str, pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza análise local do currículo sem usar IA.
    """
    # Extrair palavras-chave da vaga
    job_keywords = extract_keywords_from_job_description(job_desc)
    
    # Verificar quais palavras-chave estão no currículo
    resume_lower = resume_text.lower()
    keywords_found = [kw for kw in job_keywords if kw in resume_lower]
    keywords_missing = [kw for kw in job_keywords if kw not in resume_lower]
    
    # Calcular proporção de palavras-chave encontradas
    keywords_ratio = len(keywords_found) / len(job_keywords) if job_keywords else 0
    
    # Detectar adjetivos esponjosos
    fluffy_adjectives = detect_fluffy_adjectives(resume_text)
    
    # Detectar métricas
    metrics_count, metrics_examples = detect_metrics(resume_text)
    
    # Calcular score de prontidão
    readiness_score = calculate_readiness_score(
        keywords_ratio,
        len(fluffy_adjectives),
        metrics_count,
        pdf_metadata.get("risco_parsing", "Médio")
    )
    
    # Gerar veredito
    if readiness_score >= 8:
        veredito = "Currículo bem alinhado com a vaga. Boa presença de palavras-chave e evidências quantitativas."
    elif readiness_score >= 5:
        veredito = "Currículo moderadamente alinhado. Recomenda-se adicionar mais métricas e palavras-chave específicas."
    else:
        veredito = "Currículo com baixa aderência à vaga. Necessário revisar estruturalmente e adicionar evidências mensuráveis."
    
    return {
        "score_prontidao": readiness_score / 10.0,  # Normalizado para 0-1
        "veredito_10_segundos": veredito,
        "diagnostico_ats": {
            "risco_parsing": pdf_metadata.get("risco_parsing", "Médio"),
            "falhas_estruturais": pdf_metadata.get("falhas_estruturais", [])
        },
        "analise_palavras_chave": {
            "termos_ausentes_criticos": keywords_missing[:15],  # Limitar exibição
            "termos_encontrados": len(keywords_found),
            "total_termos_vaga": len(job_keywords),
            "orientacao_insercao": f"Adicione os {len(keywords_missing)} termos ausentes nas seções de experiência e habilidades."
        },
        "orientacao_resultados_xyz": {
            "percentual_itens_com_metricas": min(metrics_count * 10, 100),  # Estimativa simples
            "adjetivos_esponjosos_detectados": fluffy_adjectives,
            "total_metricas_encontradas": metrics_count,
            "amostra_reescrita_xyz": {
                "antes": "Sou um profissional proativo e dedicado com excelente comunicação.",
                "depois": "Liderei equipe de 5 pessoas, aumentando produtividade em 30% através de implementação de novos processos."
            } if fluffy_adjectives else None
        }
    }

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
            delta="Baseado na análise local"
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
    
    termos_encontrados = palavras_chave.get('termos_encontrados', 0)
    total_termos = palavras_chave.get('total_termos_vaga', 0)
    
    st.metric(
        label="Palavras-chave da Vaga",
        value=f"{termos_encontrados}/{total_termos}",
        delta=f"{(termos_encontrados/total_termos*100) if total_termos > 0 else 0:.0f}% encontradas"
    )
    
    termos_ausentes = palavras_chave.get('termos_ausentes_criticos', [])
    if termos_ausentes:
        st.write(f"**{len(termos_ausentes)} Termos Ausentes Críticos:**")
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
    total_metricas = xyz.get('total_metricas_encontradas', 0)
    
    st.metric(
        label="Métricas Numéricas",
        value=total_metricas,
        delta=f"Estimativa: {percentual_metricas}%"
    )
    
    adjetivos = xyz.get('adjetivos_esponjosos_detectados', [])
    if adjetivos:
        st.write(f"**{len(adjetivos)} Adjetivos Esponjosos Detectados (sem evidência factual):**")
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
    else:
        # Extrair texto do PDF
        with st.spinner("Extraindo texto do PDF..."):
            resume_text, pdf_metadata = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            # Análise local
            with st.spinner("Analisando currículo localmente..."):
                analysis = analyze_resume_locally(resume_text, job_description, pdf_metadata)
            
            # Exibir resultados
            display_results(analysis)
else:
    # Tela inicial
    st.title("🚀 ResumePilot")
    st.markdown("""
    ### Bem-vindo ao ResumePilot!
    
    Este é um sistema de análise de currículos 100% local que utiliza lógica em Python para:
    
    - ✅ Analisar a estrutura ATS do seu currículo
    - ✅ Identificar palavras-chave críticas ausentes
    - ✅ Detectar adjetivos esponjosos sem evidência factual
    - ✅ Sugerir reescritas usando a metodologia Google XYZ
    - ✅ Calcular score de prontidão baseado em métricas objetivas
    
    **Como usar:**
    1. Faça upload do seu currículo em PDF na barra lateral
    2. Cole a descrição da vaga que você deseja aplicar
    3. Clique em "Analisar Currículo"
    
    📌 **Nota:** Esta versão é 100% gratuita e não requer chaves de API. Toda a análise é feita localmente no seu navegador.
    """)
