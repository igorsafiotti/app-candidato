import streamlit as st
import fitz  # PyMuPDF
import json
from typing import Dict, Any, List, Tuple
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from hardskills_database import HARD_SKILLS_DATABASE

# Termos da whitelist que colidem com preposições, artigos, verbos ou palavras comuns em PT-BR.
AMBIGUOUS_PT_TERMS = {
    'nas', 'para', 'com', 'sem', 'até', 'sobre', 'depois', 'antes', 'entre', 'sob',
    'car', 'r', 'as', 'os', 'o', 'a', 'e', 'da', 'do', 'em', 'um', 'uma', 'se', 'na',
    'que', 'foi', 'ser', 'estar', 'ter', 'haver', 'fazer', 'ir', 'vir', 'dar', 'ver',
    'lidar', 'também', 'mais', 'muito', 'pouco', 'muitos', 'poucos', 'todo', 'toda',
    'onde', 'quando', 'como', 'por', 'pela', 'pelo', 'ao', 'à', 'às', 'aos',
    'pelas', 'pelos', 'das', 'dos', 'pode', 'são', 'sao', 'está', 'esta', 'estao',
    'estava', 'estavam', 'nos', 'no', 'me', 'te', 'lhe', 'já', 'so', 'só',
}

# Qualificadores técnicos adjacentes (1–2 tokens) que autorizam termos ambíguos/curtos.
GENERIC_TECH_QUALIFIERS = (
    'storage', 'armazenamento', 'arquivamento', 'network', 'rede', 'sensor', 'sensors',
    'sensores', 'linguagem', 'language', 'languages', 'programming', 'programação',
    'programacao', 'protocol', 'protocolo', 'framework', 'library', 'biblioteca',
    'database', 'banco', 'server', 'servidor', 'cloud', 'api', 'sistema', 'system',
    'tool', 'ferramenta', 'technology', 'tecnologia', 'scanning', 'scan', 'mapeamento',
    'laser', 'studio', 'cadastro', 'ambiental', 'rural',
)

SKILL_TECH_QUALIFIERS = {
    'nas': ('storage', 'network', 'attached', 'san', 'iscsi', 'arquivamento', 'armazenamento', 'filer', 'netapp'),
    'r': ('linguagem', 'language', 'studio', 'programming', 'programação', 'programacao', 'estatística', 'estatistica', 'tidyverse', 'ggplot', 'cran'),
    'lidar': ('sensor', 'laser', 'scanning', 'scan', 'mapeamento', 'fotogrametria', 'photogrammetry', 'point cloud', 'nuvem de pontos'),
    'car': ('cadastro', 'ambiental', 'rural', 'sicar', 'sicarb'),
    'as': ('9100', 'as9100', 'aerospace'),
}

# Configuração da página
st.set_page_config(
    page_title="ResumePilot",
    page_icon="📄",
    layout="wide"
)

# Sidebar para upload e inputs
st.sidebar.title("ResumePilot")
st.sidebar.markdown("Análise Estatística de Currículos")

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

@st.cache_data
def load_blacklist_adjetivos() -> Dict[str, Any]:
    """Carrega a blacklist de adjetivos do arquivo JSON com cache."""
    try:
        with open('blacklist_adjetivos.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Arquivo blacklist_adjetivos.json não encontrado. Usando lista vazia.")
        return {"categorias": []}
    except json.JSONDecodeError as e:
        st.error(f"Erro ao ler blacklist_adjetivos.json: {str(e)}")
        return {"categorias": []}
    except Exception as e:
        st.error(f"Erro inesperado ao carregar blacklist: {str(e)}")
        return {"categorias": []}

@st.cache_data
def load_taxonomy() -> Dict[str, Any]:
    """Carrega a taxonomia do arquivo JSON com cache."""
    try:
        with open('taxonomy.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Arquivo taxonomy.json não encontrado. Algumas funcionalidades podem estar limitadas.")
        return {"areas": [], "seniority_modifiers": []}
    except json.JSONDecodeError as e:
        st.error(f"Erro ao ler taxonomy.json: {str(e)}")
        return {"areas": [], "seniority_modifiers": []}
    except Exception as e:
        st.error(f"Erro inesperado ao carregar taxonomia: {str(e)}")
        return {"areas": [], "seniority_modifiers": []}

@st.cache_data
def load_credentials_database() -> Dict[str, Any]:
    """Carrega a base de dados de credenciais do arquivo JSON com cache."""
    try:
        with open('credentials_database.json', 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        st.warning("Arquivo credentials_database.json não encontrado. Algumas funcionalidades podem estar limitadas.")
        return {}
    except json.JSONDecodeError as e:
        st.error(f"Erro ao ler credentials_database.json: {str(e)}")
        return {}
    except Exception as e:
        st.error(f"Erro inesperado ao carregar credenciais: {str(e)}")
        return {}

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

def _skill_word_pattern(skill: str) -> str:
    """Regex com limites de palavra estritos para um termo da whitelist."""
    return rf"\b{re.escape(skill)}\b"


def _requires_guarded_skill_match(skill: str) -> bool:
    """Siglas curtas (< 4) e termos ambíguos PT-BR exigem maiúsculas ou qualificador técnico."""
    return len(skill) < 4 or skill.lower() in AMBIGUOUS_PT_TERMS


def _adjacent_qualifier_window(texto: str, start: int, end: int) -> str:
    """Retorna até 2 tokens imediatamente antes e depois da ocorrência."""
    left_tokens = re.findall(r"[\w+#.\-]+", texto[:start])[-2:]
    right_tokens = re.findall(r"[\w+#.\-]+", texto[end:])[:2]
    return " ".join(left_tokens + right_tokens).lower()


def _has_adjacent_tech_qualifier(skill: str, texto: str, start: int, end: int) -> bool:
    window = _adjacent_qualifier_window(texto, start, end)
    if not window:
        return False
    qualifiers = SKILL_TECH_QUALIFIERS.get(skill.lower(), ()) + GENERIC_TECH_QUALIFIERS
    return any(re.search(_skill_word_pattern(q), window) for q in qualifiers)


def is_valid_skill_match(skill: str, texto: str) -> bool:
    """
    Verifica se um termo técnico deve ser considerado válido no match.
    Resolve colisões entre palavras comuns em PT-BR e siglas técnicas.
    """
    if not skill or not texto:
        return False

    if _requires_guarded_skill_match(skill):
        for match in re.finditer(_skill_word_pattern(skill), texto, flags=re.IGNORECASE):
            token = match.group(0)
            if token.isupper():
                return True
            if _has_adjacent_tech_qualifier(skill, texto, match.start(), match.end()):
                return True
        return False

    return bool(re.search(_skill_word_pattern(skill.lower()), texto.lower()))


def deduplicate_skills_by_subsumption(terms: List[str]) -> List[str]:
    """
    Remove subtermos isolados quando um n-grama mais longo já os contém
    (ex.: 'licitação' some se 'processos de licitação' estiver na lista).
    """
    unique: List[str] = []
    seen = set()
    for term in terms:
        key = term.lower().strip()
        if key and key not in seen:
            unique.append(term)
            seen.add(key)

    drop_indexes = set()
    for i, shorter in enumerate(unique):
        shorter_norm = shorter.lower().strip()
        for j, longer in enumerate(unique):
            if i == j:
                continue
            longer_norm = longer.lower().strip()
            if len(longer_norm) <= len(shorter_norm):
                continue
            if re.search(_skill_word_pattern(shorter_norm), longer_norm):
                drop_indexes.add(i)
                break

    return [term for i, term in enumerate(unique) if i not in drop_indexes]


def extract_matching_skills(texto: str, catalog: List[str]) -> List[str]:
    """Extrai skills do catálogo presentes no texto, já desduplicadas por subsunção."""
    found = [skill for skill in catalog if is_valid_skill_match(skill, texto)]
    return deduplicate_skills_by_subsumption(found)

def detect_fluffy_adjectives(resume_text: str, blacklist: Dict[str, Any]) -> List[str]:
    """
    Detecta adjetivos esponjosos no currículo usando a blacklist.
    Usa word boundaries para evitar falsos positivos por substring.
    """
    found_adjectives = []
    resume_lower = resume_text.lower()
    
    for categoria in blacklist.get("categorias", []):
        for termo_data in categoria.get("termos", []):
            termo = termo_data.get("termo", "").lower()
            variacoes = [v.lower() for v in termo_data.get("variacoes", [])]
            
            # Verificar termo principal com word boundary
            if re.search(rf"\b{re.escape(termo)}\b", resume_lower):
                found_adjectives.append(termo)
            
            # Verificar variações com word boundary
            for variacao in variacoes:
                if re.search(rf"\b{re.escape(variacao)}\b", resume_lower):
                    found_adjectives.append(variacao)
    
    return list(set(found_adjectives))  # Remover duplicatas

def analyze_resume_stat(resume_text: str, job_desc: str, pdf_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Realiza análise estatística local do currículo usando TF-IDF.
    Motor NLP focado em extração inteligente de palavras-chave.
    """
    # Stopwords básicas em português (versão estendida com verbos)
    pt_stopwords_extended = [
        'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo',
        # Verbos comuns para filtrar do TF-IDF
        'acompanhar', 'garantir', 'realizar', 'atuar', 'ser', 'estar', 'ter', 'fazer', 'ir', 'vir', 'dar', 'ver', 'dizer', 'saber', 'querer', 'poder', 'dever', 'precisar', 'trabalhar', 'viver', 'ficar', 'passar', 'chegar', 'sair', 'entrar', 'sair', 'ficar', 'tornar', 'parecer', 'começar', 'terminar', 'acabar', 'conseguir', 'obter', 'levar', 'trazer', 'colocar', 'colocar', 'pôr', 'poder', 'saber', 'conhecer', 'entender', 'compreender', 'pensar', 'crer', 'achar', 'sentir', 'querer', 'desejar', 'esperar', 'procurar', 'buscar', 'encontrar', 'descobrir', 'identificar', 'reconhecer', 'perceber', 'notar', 'observar', 'analisar', 'estudar', 'aprender', 'ensinar', 'explicar', 'mostrar', 'demonstrar', 'provar', 'confirmar', 'verificar', 'validar', 'testar', 'experimentar', 'tentar', 'ajudar', 'apoiar', 'assistir', 'servir', 'atender', 'cuidar', 'tratar', 'lidar', 'resolver', 'solucionar', 'corrigir', 'melhorar', 'otimizar', 'aumentar', 'reduzir', 'crescer', 'desenvolver', 'construir', 'criar', 'produzir', 'fabricar', 'vender', 'comprar', 'adquirir', 'obter', 'ganhar', 'perder', 'ganhar', 'perder', 'receber', 'pagar', 'investir', 'economizar', 'gastar', 'desperdiçar', 'usar', 'utilizar', 'empregar', 'aplicar', 'implementar', 'executar', 'realizar', 'efetuar', 'concluir', 'finalizar', 'completar', 'iniciar', 'começar', 'iniciar', 'lançar', 'criar', 'estabelecer', 'definir', 'determinar', 'decidir', 'escolher', 'selecionar', 'preferir', 'gostar', 'amar', 'odiar', 'detestar', 'aborrecer', 'alegrar', 'comemorar', 'celebrar', 'festejar', 'agradecer', 'agradecer', 'agradecer', 'louvar', 'elogiar', 'criticar', 'julgar', 'avaliar', 'considerar', 'pensar', 'refletir', 'meditar', 'planejar', 'organizar', 'coordenar', 'gerenciar', 'administrar', 'dirigir', 'conduzir', 'liderar', 'comandar', 'controlar', 'supervisionar', 'monitorar', 'acompanhar', 'assessorar', 'aconselhar', 'orientar', 'instruir', 'treinar', 'educar', 'formar', 'preparar', 'capacitar', 'habilitar', 'possibilitar', 'permitir', 'autorizar', 'autorizar', 'permitir', 'liberar', 'libertar', 'desbloquear', 'bloquear', 'impedir', 'evitar', 'prevenir', 'proteger', 'defender', 'atacar', 'agredir', 'ofender', 'insultar', 'ofender', 'humilhar', 'degradar', 'humilhar', 'destruir', 'construir', 'reconstruir', 'reconstruir', 'transformar', 'mudar', 'alterar', 'modificar', 'adaptar', 'ajustar', 'corrigir', 'consertar', 'reparar', 'consertar', 'arrumar', 'organizar', 'limpar', 'organizar', 'arrumar', 'organizar', 'preparar', 'preparar', 'preparar'
    ]
    
    # Calcular similaridade de cosseno geral (usado como fallback e ponderação)
    try:
        vectorizer = TfidfVectorizer(
            stop_words=pt_stopwords_extended,
            ngram_range=(1, 2),
            max_features=100
        )
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except Exception as e:
        similarity = 0.0
    
    # Estratégia 1: Extrair skills da vaga usando whitelist (HARD_SKILLS_DATABASE)
    hard_skills_vaga = extract_matching_skills(job_desc, HARD_SKILLS_DATABASE)
    
    # Se encontrou skills da database na vaga, usar essas como referência
    if hard_skills_vaga:
        # Limitar a 15 skills mais relevantes (ou todas se menos que 15)
        hard_skills_vaga = hard_skills_vaga[:15]
        
        # Verificar quais dessas skills estão no currículo (usando validação refinada)
        hard_skills_encontradas = []
        hard_skills_ausentes = []
        
        for skill in hard_skills_vaga:
            if is_valid_skill_match(skill, resume_text):
                hard_skills_encontradas.append(skill)
            else:
                hard_skills_ausentes.append(skill)

        hard_skills_encontradas = deduplicate_skills_by_subsumption(hard_skills_encontradas)
        hard_skills_ausentes = deduplicate_skills_by_subsumption(hard_skills_ausentes)
        
        # Calcular score baseado em proporção de skills da vaga encontradas
        proporcao_skills = len(hard_skills_encontradas) / len(hard_skills_vaga) if hard_skills_vaga else 0
        score_prontidao = (proporcao_skills * 0.7 + similarity * 0.3) * 10
        
        metodo_usado = "whitelist_database"
    
    else:
        # Estratégia 2: Fallback - usar TF-IDF filtrado com stopwords estendidas
        try:
            vectorizer = TfidfVectorizer(
                stop_words=pt_stopwords_extended,
                ngram_range=(1, 2),
                max_features=100
            )
            tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
            
            # Extrair as Top 15 features (palavras/bigramas) com maiores pesos da vaga
            feature_names = vectorizer.get_feature_names_out()
            tfidf_scores = tfidf_matrix[1].toarray()[0]  # Scores da vaga (índice 1)
            
            # Criar lista de (feature, score) e ordenar por score
            feature_score_pairs = list(zip(feature_names, tfidf_scores))
            feature_score_pairs.sort(key=lambda x: x[1], reverse=True)
            
            # Candidatos TF-IDF, já filtrados contra colisões PT-BR e desduplicados por subsunção
            tfidf_candidates = [
                feature for feature, score in feature_score_pairs
                if score > 0 and is_valid_skill_match(feature, job_desc)
            ]
            hard_skills_vaga = deduplicate_skills_by_subsumption(tfidf_candidates)[:15]
            
            # Verificar presença no currículo (usando validação refinada para todos os termos)
            hard_skills_encontradas = []
            hard_skills_ausentes = []
            
            for feature in hard_skills_vaga:
                if is_valid_skill_match(feature, resume_text):
                    hard_skills_encontradas.append(feature)
                else:
                    hard_skills_ausentes.append(feature)

            hard_skills_encontradas = deduplicate_skills_by_subsumption(hard_skills_encontradas)
            hard_skills_ausentes = deduplicate_skills_by_subsumption(hard_skills_ausentes)
            
            # Calcular score baseado em proporção de skills encontradas
            proporcao_skills = len(hard_skills_encontradas) / len(hard_skills_vaga) if hard_skills_vaga else 0
            score_prontidao = (proporcao_skills * 0.7 + similarity * 0.3) * 10
            
            metodo_usado = "tfidf_fallback"
            
        except Exception as e:
            similarity = 0.0
            hard_skills_vaga = []
            hard_skills_encontradas = []
            hard_skills_ausentes = []
            score_prontidao = 0.0
            metodo_usado = "erro"
    
    # Skills da whitelist presentes no currículo, sem subtermos redundantes
    hard_skills_db_matches = extract_matching_skills(resume_text, HARD_SKILLS_DATABASE)
    
    # Veredito baseado no score
    if score_prontidao >= 7:
        veredito = "Currículo com boa compatibilidade estatística. As palavras-chave da vaga estão bem representadas."
    elif score_prontidao >= 4:
        veredito = "Currículo com compatibilidade moderada. Algumas palavras-chave importantes estão ausentes."
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
            "hard_skills_vaga": hard_skills_vaga,
            "hard_skills_encontradas": hard_skills_encontradas,
            "hard_skills_ausentes": hard_skills_ausentes,
            "hard_skills_db_matches": hard_skills_db_matches,
            "metodo_extracao": metodo_usado,
            "orientacao_insercao": f"Adicione os {len(hard_skills_ausentes)} termos ausentes para melhorar compatibilidade."
        }
    }

def display_results(analysis: Dict[str, Any], adjetivos_detectados: List[str]):
    """
    Exibe os resultados da análise no dashboard do Streamlit.
    """
    st.header("📊 Resultados da Análise")
    
    # Score de Prontidão
    col1, col2, col3 = st.columns(3)
    with col1:
        score_valor = min(analysis.get('score_prontidao', 0), 10.0)  # Garantir máximo de 10
        st.metric(
            label="Score de Prontidão",
            value=f"{score_valor:.1f}/10",
            delta="Análise estatística TF-IDF"
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
    st.subheader("🔑 Análise de Hard Skills")
    palavras_chave = analysis.get('analise_palavras_chave', {})
    
    hard_skills_vaga = palavras_chave.get('hard_skills_vaga', [])
    hard_skills_encontradas = palavras_chave.get('hard_skills_encontradas', [])
    hard_skills_ausentes = palavras_chave.get('hard_skills_ausentes', [])
    hard_skills_db_matches = palavras_chave.get('hard_skills_db_matches', [])
    metodo_extracao = palavras_chave.get('metodo_extracao', 'desconhecido')
    
    # Métrica clara: X/Y Hard Skills Encontradas (Y pode variar se usar whitelist)
    total_vaga = len(hard_skills_vaga) if hard_skills_vaga else 15
    st.metric(
        label="Hard Skills da Vaga",
        value=f"{len(hard_skills_encontradas)}/{total_vaga}",
        delta=f"{(len(hard_skills_encontradas)/total_vaga*100) if total_vaga > 0 else 0:.0f}% encontradas"
    )
    
    # Indicar método de extração usado
    if metodo_extracao == "whitelist_database":
        st.info("📚 **Método:** Whitelist da Database (Termos técnicos validados)")
    elif metodo_extracao == "tfidf_fallback":
        st.info("🔍 **Método:** TF-IDF Fallback (Vaga sem skills da database)")
    else:
        st.warning(f"⚠️ **Método:** {metodo_extracao}")
    
    # Exibir TODAS as Hard Skills encontradas
    if hard_skills_encontradas:
        st.write(f"**{len(hard_skills_encontradas)} Hard Skills Encontradas:**")
        for skill in hard_skills_encontradas:
            st.success(f"✓ {skill}")
    else:
        st.warning("⚠ Nenhuma Hard Skill da vaga encontrada no currículo")
    
    # Exibir TODAS as Hard Skills ausentes (sem truncar)
    if hard_skills_ausentes:
        st.write(f"**{len(hard_skills_ausentes)} Hard Skills Ausentes:**")
        for skill in hard_skills_ausentes:
            st.warning(f"• {skill}")
    else:
        st.success("✅ Todas as Hard Skills da vaga estão presentes")
    
    # Exibir skills da database local encontradas
    if hard_skills_db_matches:
        st.write(f"**{len(hard_skills_db_matches)} Skills da Database Local:**")
        for skill in hard_skills_db_matches[:20]:  # Limitar a 20 para não poluir
            st.info(f"📚 {skill}")
        if len(hard_skills_db_matches) > 20:
            st.info(f"...e mais {len(hard_skills_db_matches) - 20} skills")
    
    orientacao = palavras_chave.get('orientacao_insercao', '')
    if orientacao:
        st.info(f"**Orientação para Inserção:** {orientacao}")
    
    st.divider()
    
    # Análise de Adjetivos Esponjosos
    st.subheader("⚠️ Análise de Adjetivos Esponjosos")
    if adjetivos_detectados:
        st.write(f"**{len(adjetivos_detectados)} Adjetivos Esponjosos Detectados:**")
        for adjetivo in adjetivos_detectados:
            st.warning(f"• {adjetivo}")
        st.info("💡 **Dica:** Substitua adjetivos genéricos por exemplos concretos de ações e resultados.")
    else:
        st.success("✅ Nenhum adjetivo esponjoso detectado")

# Lógica principal
if analyze_button:
    if not uploaded_file:
        st.sidebar.error("Por favor, faça upload de um currículo em PDF.")
    elif not job_description:
        st.sidebar.error("Por favor, insira a descrição da vaga.")
    else:
        # Carregar bases de dados
        blacklist = load_blacklist_adjetivos()
        taxonomy = load_taxonomy()
        credentials = load_credentials_database()
        
        # Extrair texto do PDF (análise local)
        with st.spinner("Extraindo texto do PDF e avaliando estrutura ATS..."):
            resume_text, pdf_metadata = extract_text_from_pdf(uploaded_file)
        
        if resume_text:
            # Detectar adjetivos esponjosos
            adjetivos_detectados = detect_fluffy_adjectives(resume_text, blacklist)
            
            # Análise estatística local
            with st.spinner("Realizando Análise Estatística (TF-IDF)..."):
                analysis = analyze_resume_stat(resume_text, job_description, pdf_metadata)
            
            if analysis:
                # Exibir resultados
                display_results(analysis, adjetivos_detectados)
else:
    # Tela inicial
    st.title("🚀 ResumePilot")
    st.markdown("""
    ### Bem-vindo ao ResumePilot!
    
    Este é um sistema de análise de currículos baseado em NLP estatístico local:
    
    - ✅ **Análise ATS Local (PyMuPDF)**: Avaliação estrutural do PDF
    - ✅ **Análise Estatística (TF-IDF)**: Extração inteligente de palavras-chave com n-gramas
    - ✅ **Database Local**: Cross-check com base de Hard Skills (1000+ termos)
    - ✅ **Blacklist de Adjetivos**: Detecção de clichês e termos esponjosos
    - ✅ **Termos Compostos**: Captura bigramas como 'gestão financeira' e 'engenharia civil'
    - ✅ **Score de Prontidão**: Nota baseada em proporção de skills e similaridade de texto
    - ✅ **100% Local**: Rápido, sem API Key, sem custos
    
    **Como usar:**
    1. Faça upload do seu currículo em PDF na barra lateral
    2. Cole a descrição da vaga que você deseja aplicar
    3. Clique em "Analisar Currículo"
    
    🔒 **Privacidade:** Toda a análise é feita localmente. Nenhum dado é enviado para servidores externos.
    """)
