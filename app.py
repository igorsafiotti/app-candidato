import json
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import fitz  # PyMuPDF
import streamlit as st
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from hardskills_database import HARD_SKILLS_DATABASE

BASE_DIR = Path(__file__).resolve().parent

# Configuração da página Streamlit
st.set_page_config(
    page_title="ResumePilot AI - Diagnóstico ATS",
    page_icon="📄",
    layout="wide"
)

# Termos da whitelist que colidem com preposições, artigos, verbos ou palavras comuns em PT-BR
AMBIGUOUS_PT_TERMS = {
    'nas', 'para', 'com', 'sem', 'até', 'sobre', 'depois', 'antes', 'entre', 'sob',
    'car', 'r', 'as', 'os', 'o', 'a', 'e', 'da', 'do', 'em', 'um', 'uma', 'se', 'na',
    'que', 'foi', 'ser', 'estar', 'ter', 'haver', 'fazer', 'ir', 'vir', 'dar', 'ver',
    'lidar', 'também', 'mais', 'muito', 'pouco', 'muitos', 'poucos', 'todo', 'toda',
    'onde', 'quando', 'como', 'por', 'pela', 'pelo', 'ao', 'à', 'às', 'aos',
    'pelas', 'pelos', 'das', 'dos', 'pode', 'são', 'sao', 'está', 'esta', 'estao',
    'estava', 'estavam', 'nos', 'no', 'me', 'te', 'lhe', 'já', 'so', 'só',
}

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

PT_STOPWORDS_EXTENDED = [
    'de', 'a', 'o', 'que', 'e', 'do', 'da', 'em', 'um', 'para', 'com', 'não', 'uma', 'os', 'no', 'se', 'na', 'por', 'mais', 'as', 'dos', 'como', 'mas', 'ao', 'ele', 'das', 'seu', 'sua', 'ou', 'quando', 'muito', 'nos', 'já', 'eu', 'também', 'só', 'pelo', 'pela', 'até', 'isso', 'ela', 'entre', 'depois', 'sem', 'mesmo', 'aos', 'seus', 'quem', 'nas', 'me', 'esse', 'eles', 'você', 'essa', 'num', 'nem', 'suas', 'meu', 'às', 'minha', 'numa', 'pelos', 'elas', 'qual', 'nós', 'lhe', 'deles', 'essas', 'esses', 'pelas', 'este', 'dele', 'tu', 'te', 'vocês', 'vos', 'lhes', 'meus', 'minhas', 'teu', 'tua', 'teus', 'tuas', 'nosso', 'nossa', 'nossos', 'nossas', 'dela', 'delas', 'esta', 'estes', 'estas', 'aquele', 'aquela', 'aqueles', 'aquelas', 'isto', 'aquilo',
    'acompanhar', 'garantir', 'realizar', 'atuar', 'ser', 'estar', 'ter', 'fazer', 'ir', 'vir', 'dar', 'ver', 'dizer', 'saber', 'querer', 'poder', 'dever', 'precisar', 'trabalhar', 'viver', 'ficar', 'passar', 'chegar', 'sair', 'entrar', 'tornar', 'parecer', 'começar', 'terminar', 'acabar', 'conseguir', 'obter', 'levar', 'trazer', 'colocar', 'pôr', 'conhecer', 'entender', 'compreender', 'pensar', 'crer', 'achar', 'sentir', 'desejar', 'esperar', 'procurar', 'buscar', 'encontrar', 'descobrir', 'identificar', 'reconhecer', 'perceber', 'notar', 'observar', 'analisar', 'estudar', 'aprender', 'ensinar', 'explicar', 'mostrar', 'demonstrar', 'provar', 'confirmar', 'verificar', 'validar', 'testar', 'experimentar', 'tentar', 'ajudar', 'apoiar', 'assistir', 'servir', 'atender', 'cuidar', 'tratar', 'lidar', 'resolver', 'solucionar', 'corrigir', 'melhorar', 'otimizar', 'aumentar', 'reduzir', 'crescer', 'desenvolver', 'construir', 'criar', 'produzir', 'fabricar', 'vender', 'comprar', 'adquirir', 'ganhar', 'perder', 'receber', 'pagar', 'investir', 'economizar', 'gastar', 'usar', 'utilizar', 'empregar', 'aplicar', 'implementar', 'executar', 'efetuar', 'concluir', 'finalizar', 'completar', 'iniciar', 'lançar', 'estabelecer', 'definir', 'determinar', 'decidir', 'escolher', 'selecionar', 'preferir', 'gostar', 'amar', 'odiar', 'planejar', 'organizar', 'coordenar', 'gerenciar', 'administrar', 'dirigir', 'conduzir', 'liderar', 'comandar', 'controlar', 'supervisionar', 'monitorar', 'assessorar', 'aconselhar', 'orientar', 'instruir', 'treinar', 'educar', 'formar', 'preparar', 'capacitar',
]

# Dicionário de equivalências práticas para o ATS
EQUIVALENCIAS_SKILLS = {
    'ga4': ['google analytics', 'ga4', 'google analytics 4', 'analytics'],
    'metodologias ágeis': ['agil', 'agile', 'scrum', 'kanban', 'metodologias ageis', 'gestao agil', 'projetos ageis'],
    'scrum': ['scrum', 'sprint', 'sprints', 'scrum master', 'daily'],
    'kanban': ['kanban', 'quadro kanban', 'gestao visual'],
    'gestão de projetos': ['gestao de projetos', 'project management', 'gerenciamento de projetos', 'gestao de portfolio', 'coordenador de projetos', 'coordenacao de projetos', 'digital project'],
    'looker': ['looker studio', 'looker', 'google data studio', 'data studio'],
}

# ==============================================================================
# CARREGAMENTO DE BASES LOCAIS (CACHE)
# ==============================================================================
def _load_json_file(filename: str, default: Any) -> Any:
    path = BASE_DIR / filename
    try:
        with open(path, 'r', encoding='utf-8') as handle:
            return json.load(handle)
    except FileNotFoundError:
        st.warning(f"Ficheiro {filename} não encontrado. Algumas funcionalidades podem estar limitadas.")
        return default
    except json.JSONDecodeError as exc:
        st.error(f"Erro ao ler {filename}: {exc}")
        return default
    except Exception as exc:
        st.error(f"Erro inesperado ao carregar {filename}: {exc}")
        return default

@st.cache_data
def load_blacklist_adjetivos() -> Dict[str, Any]:
    return _load_json_file('blacklist_adjetivos.json', {"categorias": []})

@st.cache_data
def load_taxonomy() -> Dict[str, Any]:
    return _load_json_file('taxonomy.json', {"areas": [], "seniority_modifiers": []})

@st.cache_data
def load_credentials_database() -> Dict[str, Any]:
    return _load_json_file('credentials_database.json', {"categorias": []})

@st.cache_data
def load_action_verbs() -> Dict[str, Any]:
    return _load_json_file('executive_action_verbs.json', {"categories": [], "weak_terms_alerts": []})

# ==============================================================================
# NORMALIZAÇÃO E MATCHING REGEX DETERMINÍSTICO
# ==============================================================================
def _fold_accents(texto: str) -> str:
    normalized = unicodedata.normalize('NFKD', texto)
    return ''.join(char for char in normalized if not unicodedata.combining(char))

def _skill_word_pattern(skill: str) -> str:
    return rf"\b{re.escape(skill)}\b"

def _requires_guarded_skill_match(skill: str) -> bool:
    return len(skill) < 4 or skill.lower() in AMBIGUOUS_PT_TERMS

def _adjacent_qualifier_window(texto: str, start: int, end: int) -> str:
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

def _count_whole_word(term: str, texto: str) -> int:
    if not term or not texto:
        return 0
    return len(re.findall(_skill_word_pattern(_fold_accents(term.strip())), _fold_accents(texto), flags=re.IGNORECASE))

def _term_in_text(term: str, texto: str) -> bool:
    return _count_whole_word(term, texto) > 0

def normalizar_skill_match(skill: str, texto: str) -> bool:
    """Verifica correspondência direta ou por equivalências práticas."""
    skill_lower = skill.lower().strip()
    if skill_lower in EQUIVALENCIAS_SKILLS:
        for sinom in EQUIVALENCIAS_SKILLS[skill_lower]:
            if _term_in_text(sinom, texto):
                return True
    return is_valid_skill_match(skill, texto)

def deduplicate_skills_by_subsumption(terms: List[str]) -> List[str]:
    """Elimina subtermos isolados contidos em n-gramas maiores (O(N log N))."""
    unique: List[str] = []
    seen = set()
    for term in terms:
        key = term.lower().strip()
        if key and key not in seen:
            unique.append(term)
            seen.add(key)

    ranked = sorted(unique, key=lambda item: len(item.strip()), reverse=True)
    accepted: List[str] = []
    accepted_normalized: List[str] = []
    kept_keys = set()

    for term in ranked:
        normalized = term.lower().strip()
        pattern = re.compile(_skill_word_pattern(normalized))
        if any(pattern.search(longer) for longer in accepted_normalized):
            continue
        accepted.append(term)
        accepted_normalized.append(normalized)
        kept_keys.add(normalized)

    return [term for term in unique if term.lower().strip() in kept_keys]

# ==============================================================================
# EXTRAÇÃO DE TEXTO DO PDF (ORDENAÇÃO COLUNAR)
# ==============================================================================
def extract_text_from_pdf(pdf_file) -> Tuple[str, Dict[str, Any]]:
    try:
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)
        pdf_bytes = pdf_file.read()
        if hasattr(pdf_file, 'seek'):
            pdf_file.seek(0)

        pdf_document = fitz.open(stream=pdf_bytes, filetype="pdf")
        page_texts: List[str] = []
        total_pages = len(pdf_document)
        pages_with_text = 0
        total_blocks = 0
        text_blocks_count = 0

        for page in pdf_document:
            raw_blocks = page.get_text("blocks") or []
            total_blocks += len(raw_blocks)
            valid_blocks = [
                block for block in raw_blocks
                if isinstance(block, (list, tuple)) and len(block) > 6 and block[6] == 0
            ]
            # Ordena por coluna horizontal (aproximação a 50px) e depois verticalmente
            valid_blocks.sort(key=lambda b: (round(b[0] / 50) * 50, b[1]))
            text_blocks_count += len(valid_blocks)
            page_text = "\n".join(str(block[4]).strip() for block in valid_blocks if str(block[4]).strip())
            if page_text:
                pages_with_text += 1
                page_texts.append(page_text)

        pdf_document.close()
        text = "\n\n".join(page_texts)

        parsing_risk = "Baixo"
        structural_issues: List[str] = []
        if pages_with_text < total_pages:
            parsing_risk = "Médio"
            structural_issues.append(f"{total_pages - pages_with_text} página(s) sem texto detetável")
        
        # Ajuste de sensibilidade: Layouts normais de 2 colunas têm entre 20 e 35 blocos por página
        if total_pages and total_blocks > total_pages * 45:
            parsing_risk = "Alto"
            structural_issues.append("Complexidade excessiva de blocos (risco de sobreposição de texto)")
        elif total_pages and total_blocks > total_pages * 15:
            parsing_risk = "Baixo (Layout em Colunas Detetado)"

        return text, {
            "risco_parsing": parsing_risk,
            "falhas_estruturais": structural_issues,
            "paginas_com_texto": pages_with_text,
            "total_paginas": total_pages,
            "blocos_texto": text_blocks_count,
        }
    except Exception as exc:
        st.error(f"Erro ao extrair texto do PDF: {exc}")
        return "", {
            "risco_parsing": "Alto",
            "falhas_estruturais": [str(exc)],
            "paginas_com_texto": 0,
            "total_paginas": 0,
            "blocos_texto": 0,
        }

# ==============================================================================
# AUDITORIAS DETERMINÍSTICAS (ESTILO JOBSCAN)
# ==============================================================================
def auditar_contato_e_presenca(texto_cv: str) -> Dict[str, Any]:
    """Valida a presença de dados de contacto no documento inteiro sem truncamento."""
    padrao_email = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
    tem_email = bool(re.search(padrao_email, texto_cv))

    padrao_fone = r'(?:\+?55\s?)?(?:\(?\d{2}\)?\s?)?(?:9\s?)?\d{4}[-.\s]?\d{4}'
    tem_fone = bool(re.search(padrao_fone, texto_cv))

    padrao_web = r'(?:linkedin\.com\/in\/|linkedin\.com\/|github\.com\/|https?:\/\/)'
    tem_web = bool(re.search(padrao_web, texto_cv, re.IGNORECASE))

    padrao_loc = r'\b(São Paulo|Rio de Janeiro|Curitiba|Belo Horizonte|Porto Alegre|Brasília|Campinas|Salvador|Recife|Fortaleza|SP|RJ|PR|MG|DF|RS)\b'
    tem_loc = bool(re.search(padrao_loc, _fold_accents(texto_cv), re.IGNORECASE))

    return {
        "email": tem_email,
        "telefone": tem_fone,
        "web": tem_web,
        "localizacao": tem_loc,
        "conformidade_total": tem_email and tem_fone and tem_loc
    }

def auditar_secoes_cv(texto_cv: str) -> Dict[str, Any]:
    """Identifica se as secções essenciais de triagem ATS estão presentes."""
    cv_fold = _fold_accents(texto_cv).lower()
    
    secoes_alvo = {
        "Resumo Profissional": [r'\bresumo\b', r'\bperfil\b', r'\bsobre mim\b', r'\bsummary\b', r'\bapresentacao\b'],
        "Experiência Profissional": [r'\bexperiencia(s)?\b', r'\bhistorico profissional\b', r'\batua[çc]ao\b', r'\bwork experience\b'],
        "Formação Académica": [r'\bformacao\b', r'\beducacao\b', r'\bgraduacao\b', r'\bacademico\b', r'\beducation\b'],
        "Competências Técnicas": [r'\bcompetencias\b', r'\bhabilidades\b', r'\bskills\b', r'\bconhecimentos\b', r'\bferramentas\b']
    }
    
    status_secoes = {}
    for secao, padroes in secoes_alvo.items():
        status_secoes[secao] = any(re.search(p, cv_fold, re.IGNORECASE) for p in padroes)
        
    return {
        "secoes": status_secoes,
        "todas_presentes": status_secoes["Experiência Profissional"] and status_secoes["Formação Académica"]
    }

def auditar_metricas_e_volume(texto_cv: str) -> Dict[str, Any]:
    """Audita o volume de palavras e a densidade de resultados mensuráveis."""
    palavras = texto_cv.split()
    total_palavras = len(palavras)
    
    padrao_metricas = r'(\b\d+[\.,]?\d*\s?%|\bR\$\s?\d+[\d.,]*|\$\s?\d+[\d.,]*|\b\d+\s?(?:milhões|mil|k|M|bi)\b|\b\d+\s?(?:projetos|anos|clientes|sprints|pessoas|profissionais)\b)'
    metricas_encontradas = re.findall(padrao_metricas, texto_cv, re.IGNORECASE)
    
    padrao_datas = r'(\b\d{2}/\d{4}\b|\b(?:Jan|Fev|Mar|Abr|Mai|Jun|Jul|Ago|Set|Out|Nov|Dez)[a-z]*\.?\s?\d{4}\b|\b\d{4}\s?[-–—]\s?(?:\d{4}|atual|presente)\b)'
    datas_detectadas = len(re.findall(padrao_datas, texto_cv, re.IGNORECASE))
    
    return {
        "total_palavras": total_palavras,
        "volume_adequado": 350 <= total_palavras <= 1100,
        "total_metricas": len(metricas_encontradas),
        "datas_padronizadas": datas_detectadas >= 2
    }

def auditar_kill_questions(texto_cv: str, texto_vaga: str, base_credentials: Dict[str, Any]) -> Dict[str, Any]:
    """Cruza credenciais mandatórias regulamentadas (CREA, CAU, OAB, etc.)."""
    def _credential_terms(nome_oficial: str, variacoes: List[str]) -> List[str]:
        terms = [nome_oficial, *variacoes]
        prefixes = ('graduacao em ', 'bacharel em ', 'certificacao ', 'certificado ', 'registro no ', 'registro na ')
        folded = _fold_accents(nome_oficial).lower()
        for prefix in prefixes:
            if folded.startswith(prefix):
                remainder = _fold_accents(nome_oficial)[len(prefix):].strip(' :-')
                if remainder:
                    terms.append(remainder)
        eng = re.search(r'(?i)engenharia\s+(.+)$', _fold_accents(nome_oficial))
        if eng:
            esp = eng.group(1).strip()
            terms.extend([f'Engenheiro {esp}', f'Engenheira {esp}'])
        return list(set(terms))

    exigencias = []
    status_aprovado = True

    for categoria in base_credentials.get("categorias", []):
        cat_nome = categoria.get("categoria_nome", categoria.get("categoria_id", ""))
        for cred in categoria.get("credenciais", []):
            nome_oficial = str(cred.get("nome_oficial", "")).strip()
            if not nome_oficial:
                continue
            variacoes = [str(v).strip() for v in cred.get("siglas_ou_variacoes", []) if str(v).strip()]
            termos_busca = _credential_terms(nome_oficial, variacoes)
            
            if any(_term_in_text(t, texto_vaga) for t in termos_busca):
                evidencias = [t for t in termos_busca if _term_in_text(t, texto_cv)]
                peso = str(cred.get("peso_eliminatorio", "")).strip()
                atendida = bool(evidencias)
                
                if not atendida and peso.lower() == "alto":
                    status_aprovado = False

                exigencias.append({
                    "nome_oficial": nome_oficial,
                    "categoria": cat_nome,
                    "peso_eliminatorio": peso,
                    "status": "atendida" if atendida else "reprovada",
                    "evidencias_cv": evidencias,
                    "descricao_contextual": cred.get("descricao_contextual", ""),
                })

    faltantes = [item for item in exigencias if item["status"] == "reprovada"]
    return {
        "status_aprovado": status_aprovado,
        "tem_exigencias": bool(exigencias),
        "exigencias": exigencias,
        "faltantes": faltantes,
        "atendidas": [item for item in exigencias if item["status"] == "atendida"],
    }

def avaliar_vigor_executivo(texto_cv: str, base_action_verbs: Dict[str, Any]) -> Dict[str, Any]:
    """Calcula o índice de vigor executivo cruzando verbos fortes contra termos passivos."""
    categorias = base_action_verbs.get("active_verbs") or base_action_verbs.get("categories") or []
    verbos_encontrados = []
    
    for categoria in categorias:
        cat_nome = categoria.get("category_name", categoria.get("category_id", ""))
        for verb in categoria.get("verbs", []):
            inf = str(verb.get("verb_infinitive", "")).strip()
            conj = str(verb.get("conjugated_form", "")).strip()
            forms = [f for f in (inf, conj) if f]
            hits = sum(_count_whole_word(f, texto_cv) for f in forms)
            if hits > 0:
                verbos_encontrados.append({
                    "verbo": conj if conj and _term_in_text(conj, texto_cv) else inf,
                    "infinitivo": inf,
                    "ocorrencias": hits,
                    "categoria": cat_nome
                })

    alertas = []
    for alerta in base_action_verbs.get("weak_terms_alerts", []):
        termo = str(alerta.get("weak_term", "")).strip()
        if not termo:
            continue
        variantes = [p.strip() for p in re.split(r"\s*/\s*", termo) if p.strip()]
        hits = sum(_count_whole_word(v, texto_cv) for v in variantes)
        if hits > 0:
            alertas.append({
                "weak_term": termo,
                "ocorrencias": hits,
                "issue_identified": alerta.get("issue_identified", ""),
                "suggested_replacements": alerta.get("suggested_replacements", [])
            })

    unique_verbs = len({item["infinitivo"] for item in verbos_encontrados})
    weak_total = sum(item["ocorrencias"] for item in alertas)
    
    score = int(max(0, min(100, 20 + (unique_verbs * 8) - (weak_total * 12))))

    return {
        "score_vigor": score,
        "verbos_encontrados": sorted(verbos_encontrados, key=lambda x: x["ocorrencias"], reverse=True),
        "alertas_termos_fracos": sorted(alertas, key=lambda x: x["ocorrencias"], reverse=True),
        "total_verbos_unicos": unique_verbs,
        "total_ocorrencias_fracas": weak_total
    }

def detect_fluffy_adjectives(resume_text: str, blacklist: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Deteta clichês e buzzwords devolvendo justificativa e sugestão de métrica."""
    detected = []
    for categoria in blacklist.get("categorias", []):
        for termo_data in categoria.get("termos", []):
            canonical = str(termo_data.get("termo", "")).strip()
            if not canonical:
                continue
            variants = [canonical] + [str(item).strip() for item in termo_data.get("variacoes", []) if str(item).strip()]
            total = 0
            matched_form = canonical
            for variant in variants:
                hits = _count_whole_word(variant, resume_text)
                if hits > 0 and total == 0:
                    matched_form = variant
                total += hits
            if total > 0:
                detected.append({
                    "termo_encontrado": matched_form,
                    "ocorrencias": total,
                    "justificativa": termo_data.get("justificativa", ""),
                    "substituicao_sugerida": termo_data.get("substituicao_sugerida", "")
                })

    detected.sort(key=lambda x: x["ocorrencias"], reverse=True)
    return detected

def extrair_skills_vaga_ranqueadas(texto_vaga: str, texto_cv: str, catalogo_skills: List[str]) -> List[Dict[str, Any]]:
    """Extrai hard skills da vaga ranqueando-as pela frequência e aplicando equivalências."""
    candidatos_vaga = [skill for skill in catalogo_skills if is_valid_skill_match(skill, texto_vaga)]
    candidatos_vaga = deduplicate_skills_by_subsumption(candidatos_vaga)
    
    skills_analise = []
    for skill in candidatos_vaga:
        qtd_vaga = _count_whole_word(skill, texto_vaga)
        
        # Usa correspondência por sinónimos e equivalências
        presente = normalizar_skill_match(skill, texto_cv)
        qtd_cv = _count_whole_word(skill, texto_cv) if presente else 0
        if presente and qtd_cv == 0:
            qtd_cv = 1  # Atendido por termo equivalente
        
        impacto = "Alto" if qtd_vaga >= 3 else ("Médio" if qtd_vaga == 2 else "Baixo")
        skills_analise.append({
            "skill": skill,
            "frequencia_vaga": qtd_vaga,
            "frequencia_cv": qtd_cv,
            "presente_no_cv": presente,
            "impacto": impacto
        })
        
    skills_analise.sort(key=lambda x: (x["frequencia_vaga"], x["frequencia_cv"]), reverse=True)
    return skills_analise

# ==============================================================================
# PIPELINE ESTATÍSTICO GERAL E CÁLCULO DE SCORE
# ==============================================================================
def processar_diagnostico_completo(
    resume_text: str,
    job_desc: str,
    pdf_metadata: Dict[str, Any],
    credentials: Dict[str, Any],
    action_verbs: Dict[str, Any],
    blacklist: Dict[str, Any]
) -> Dict[str, Any]:
    
    # 1. Auditorias estruturais e comportamentais
    contatos = auditar_contato_e_presenca(resume_text)
    secoes = auditar_secoes_cv(resume_text)
    metricas = auditar_metricas_e_volume(resume_text)
    kill_q = auditar_kill_questions(resume_text, job_desc, credentials)
    vigor = avaliar_vigor_executivo(resume_text, action_verbs)
    buzzwords = detect_fluffy_adjectives(resume_text, blacklist)
    
    # 2. Similaridade Cosseno TF-IDF
    try:
        vectorizer = TfidfVectorizer(stop_words=PT_STOPWORDS_EXTENDED, ngram_range=(1, 2), max_features=100)
        tfidf_matrix = vectorizer.fit_transform([resume_text, job_desc])
        similarity = float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])
    except Exception:
        similarity = 0.0

    # 3. Hard skills ranqueadas por repetição na vaga
    skills_ranqueadas = extrair_skills_vaga_ranqueadas(job_desc, resume_text, HARD_SKILLS_DATABASE)
    
    # Cálculo ponderado da proporção de skills
    if skills_ranqueadas:
        pontos_skills = sum(
            (3 if s["impacto"] == "Alto" else (2 if s["impacto"] == "Médio" else 1))
            for s in skills_ranqueadas if s["presente_no_cv"]
        )
        total_pontos_possiveis = sum(
            (3 if s["impacto"] == "Alto" else (2 if s["impacto"] == "Médio" else 1))
            for s in skills_ranqueadas
        )
        prop_skills = pontos_skills / total_pontos_possiveis if total_pontos_possiveis > 0 else 0.0
    else:
        prop_skills = 0.0

    # 4. Cálculo do Score de Prontidão (0 a 10)
    score_skills = prop_skills * 4.5                                      # 45% peso
    score_similaridade = similarity * 1.5                                  # 15% peso
    score_vigor_peso = (vigor["score_vigor"] / 100.0) * 1.5                # 15% peso
    score_metricas = min(1.0, (metricas["total_metricas"] / 5.0)) * 1.0    # 10% peso
    score_contato = (sum([contatos["email"], contatos["telefone"], contatos["web"], contatos["localizacao"]]) / 4.0) * 0.75 # 7.5% peso
    score_secoes = (sum(secoes["secoes"].values()) / 4.0) * 0.75          # 7.5% peso

    score_final = score_skills + score_similaridade + score_vigor_peso + score_metricas + score_contato + score_secoes

    # BLOQUEIO DETERMINÍSTICO: Se reprovou em Kill Question de peso ALTO, trava o score em no máximo 3.0
    status_eliminado = False
    if kill_q["faltantes"] and not kill_q["status_aprovado"]:
        score_final = min(score_final, 3.0)
        status_eliminado = True

    return {
        "score_final": min(score_final, 10.0),
        "status_eliminado": status_eliminado,
        "similaridade_tfidf": similarity,
        "contatos": contatos,
        "secoes": secoes,
        "metricas": metricas,
        "kill_questions": kill_q,
        "vigor": vigor,
        "buzzwords": buzzwords,
        "skills_ranqueadas": skills_ranqueadas,
        "pdf_metadata": pdf_metadata
    }

# ==============================================================================
# INTERFACE STREAMLIT
# ==============================================================================
st.sidebar.title("ResumePilot AI")
st.sidebar.markdown("### Motor de Triagem Determinístico (ATS)")

uploaded_file = st.sidebar.file_uploader("Upload do Currículo (PDF)", type=['pdf'])
job_description = st.sidebar.text_area("Descrição da Vaga", height=300, placeholder="Cole a descrição da vaga aqui...")
analyze_button = st.sidebar.button("Analisar Currículo", type="primary")

if analyze_button:
    if not uploaded_file:
        st.sidebar.error("Por favor, carregue um currículo em formato PDF.")
    elif not job_description:
        st.sidebar.error("Por favor, cole a descrição da vaga.")
    else:
        blacklist = load_blacklist_adjetivos()
        credentials = load_credentials_database()
        action_verbs = load_action_verbs()

        with st.spinner("A processar documento PDF e a cruzar dados..."):
            resume_text, pdf_meta = extract_text_from_pdf(uploaded_file)

        if not resume_text.strip():
            st.error("Não foi possível extrair texto do PDF. O documento pode ser uma imagem digitalizada.")
        else:
            diag = processar_diagnostico_completo(
                resume_text, job_description, pdf_meta, credentials, action_verbs, blacklist
            )

            # Cabeçalho de Resultados
            st.header("📊 Diagnóstico Global de Triagem ATS")

            col_s1, col_s2, col_s3, col_s4 = st.columns(4)
            with col_s1:
                st.metric("Score de Compatibilidade", f"{diag['score_final']:.1f}/10")
            with col_s2:
                total_k = len(diag['skills_ranqueadas'])
                enc_k = sum(1 for s in diag['skills_ranqueadas'] if s['presente_no_cv'])
                st.metric("Hard Skills da Vaga", f"{enc_k}/{total_k}")
            with col_s3:
                st.metric("Score de Vigor", f"{diag['vigor']['score_vigor']}/100")
            with col_s4:
                st.metric("Similaridade TF-IDF", f"{diag['similaridade_tfidf']*100:.0f}%")

            if diag["status_eliminado"]:
                st.error("🚨 **STATUS: DESCLASSIFICAÇÃO IMINENTE NO ATS** — O currículo não apresenta as credenciais ou conselhos obrigatórios (Kill Questions) exigidos pela vaga.")
            elif diag["score_final"] >= 8.0:
                st.success("✅ **STATUS: PRONTO PARA APLICAR** — Elevada aderência técnica, estrutural e semântica com a vaga.")
            elif diag["score_final"] >= 5.0:
                st.warning("⚠️ **STATUS: REQUER AJUSTES** — O perfil possui sinergia, mas faltam termos técnicos críticos ou métricas de impacto.")
            else:
                st.error("❌ **STATUS: ALTO RISCO DE REPROVAÇÃO** — Distante dos requisitos da vaga ou com falhas de formatação/conteúdo.")

            st.divider()

            # Estrutura em Abas
            tab_skills, tab_knockout, tab_recruiter, tab_buzzwords, tab_parsing = st.tabs([
                "🔑 Comparação de Skills",
                "🚨 Kill Questions (Eliminatório)",
                "💡 Dicas de Recrutador",
                "🧹 Termos Vagos & Buzzwords",
                "📄 Estrutura do PDF"
            ])

            # ABA 1: SKILLS COM COMPARAÇÃO DE FREQUÊNCIA
            with tab_skills:
                st.subheader("Correspondência de Hard Skills com a Vaga")
                if not diag["skills_ranqueadas"]:
                    st.info("Nenhuma hard skill da base técnica foi localizada no texto da vaga.")
                else:
                    c1, c2, c3, c4 = st.columns([3, 2, 2, 2])
                    c1.markdown("**Competência (Hard Skill)**")
                    c2.markdown("**Frequência na Vaga**")
                    c3.markdown("**Status no CV**")
                    c4.markdown("**Impacto no Score**")
                    st.divider()

                    for item in diag["skills_ranqueadas"]:
                        col1, col2, col3, col4 = st.columns([3, 2, 2, 2])
                        col1.write(f"**{item['skill']}**")
                        col2.write(f"{item['frequencia_vaga']}x no anúncio")
                        if item["presente_no_cv"]:
                            col3.success(f"✓ Presente ({item['frequencia_cv']}x)")
                        else:
                            col3.error("✗ Ausente")
                        
                        if item["impacto"] == "Alto":
                            col4.markdown("🔴 **Alto**")
                        elif item["impacto"] == "Médio":
                            col4.markdown("🟡 **Médio**")
                        else:
                            col4.markdown("⚪ Baixo")

            # ABA 2: KILL QUESTIONS
            with tab_knockout:
                st.subheader("Auditoria de Requisitos Mandatórios (Kill Questions)")
                kq = diag["kill_questions"]
                if not kq["tem_exigencias"]:
                    st.success("✅ Nenhuma credencial mandatória estrita (CREA, CAU, OAB, etc.) foi identificada nesta vaga.")
                else:
                    if kq["faltantes"]:
                        st.error("Foram identificadas credenciais mandatórias ausentes no seu currículo:")
                        for f in kq["faltantes"]:
                            st.markdown(f"- **{f['nome_oficial']}** ({f['categoria']}) — *Exigência de Peso {f['peso_eliminatorio']}*")
                            if f["descricao_contextual"]:
                                st.caption(f["descricao_contextual"])
                    if kq["atendidas"]:
                        st.success("Credenciais identificadas com sucesso no currículo:")
                        for a in kq["atendidas"]:
                            st.markdown(f"- ✓ **{a['nome_oficial']}**")

            # ABA 3: DICAS DE RECRUTADOR (RECRUITER TIPS)
            with tab_recruiter:
                st.subheader("Auditoria Estrutural de Recrutador")

                # Dados de Contato
                st.markdown("#### 1. Informações de Contato e Presença Web")
                ct = diag["contatos"]
                c_ok1, c_ok2, c_ok3, c_ok4 = st.columns(4)
                c_ok1.write(f"{'✅' if ct['email'] else '❌'} E-mail")
                c_ok2.write(f"{'✅' if ct['telefone'] else '❌'} Telefone")
                c_ok3.write(f"{'✅' if ct['web'] else '❌'} LinkedIn / GitHub")
                c_ok4.write(f"{'✅' if ct['localizacao'] else '❌'} Localização / Cidade")

                st.divider()

                # Secções do Currículo
                st.markdown("#### 2. Cabeçalhos de Secções Obrigatórias")
                sec = diag["secoes"]["secoes"]
                s_ok1, s_ok2, s_ok3, s_ok4 = st.columns(4)
                s_ok1.write(f"{'✅' if sec['Resumo Profissional'] else '❌'} Resumo")
                s_ok2.write(f"{'✅' if sec['Experiência Profissional'] else '❌'} Experiência")
                s_ok3.write(f"{'✅' if sec['Formação Académica'] else '❌'} Formação")
                s_ok4.write(f"{'✅' if sec['Competências Técnicas'] else '❌'} Competências")

                st.divider()

                # Métricas e Palavras
                st.markdown("#### 3. Métricas de Impacto e Volume")
                met = diag["metricas"]
                m_c1, m_c2, m_c3 = st.columns(3)
                with m_c1:
                    status_w = "✅ Adequado" if met["volume_adequado"] else "⚠️ Ajustar"
                    st.metric("Contagem de Palavras", f"{met['total_palavras']} palavras", delta=status_w)
                    st.caption("Ideal: entre 350 e 1.100 palavras.")
                with m_c2:
                    st.metric("Resultados Mensuráveis", f"{met['total_metricas']} identificados")
                    st.caption("Ideal: no mínimo 5 métricas numéricas (% ou valores monetários).")
                with m_c3:
                    st.metric("Padrão de Datas", "✅ Correto" if met["datas_padronizadas"] else "⚠️ Inconsistente")
                    st.caption("Datas estruturadas no padrão MM/AAAA ou AAAA - AAAA.")

                st.divider()

                # Vigor Executivo
                st.markdown("#### 4. Vigor Executivo e Termos Passivos")
                vg = diag["vigor"]
                if vg["alertas_termos_fracos"]:
                    st.warning(f"Foram identificadas {vg['total_ocorrencias_fracas']} ocorrências de linguagem passiva/subalterna:")
                    for alerta in vg["alertas_termos_fracos"]:
                        substitutos = ", ".join(alerta.get("suggested_replacements", []))
                        st.markdown(f"- **'{alerta['weak_term']}'** ({alerta['ocorrencias']}x): *{alerta['issue_identified']}*")
                        if substitutos:
                            st.caption(f"💡 Sugestão de substituição por verbos fortes: **{substitutos}**")
                else:
                    st.success("✅ Excelente protagonismo: nenhum termo passivo identificado.")

            # ABA 4: ADJETIVOS ESPONJOSOS (BUZZWORDS)
            with tab_buzzwords:
                st.subheader("Adjetivos Subjetivos e Buzzwords Detetadas")
                bz = diag["buzzwords"]
                if not bz:
                    st.success("✅ Nenhum clichê corporativo ou adjetivo esponjoso detetado. Discurso limpo e factual.")
                else:
                    st.write(f"Foram detetados **{len(bz)} termos vagos**. O ATS penaliza autoelogios que carecem de evidências práticas:")
                    for adjetivo in bz:
                        with st.expander(f"⚠️ '{adjetivo['termo_encontrado']}' ({adjetivo['ocorrencias']}x no documento)"):
                            st.markdown(f"**Por que o ATS desvaloriza:** {adjetivo['justificativa']}")
                            st.info(f"**Como reescrever:** {adjetivo['substituicao_sugerida']}")

            # ABA 5: PARSING DO PDF
            with tab_parsing:
                st.subheader("Auditoria de Legibilidade do PDF (PyMuPDF)")
                meta = diag["pdf_metadata"]
                p_c1, p_c2 = st.columns(2)
                p_c1.metric("Risco de Parsing", meta.get("risco_parsing", "Baixo"))
                p_c2.metric("Páginas Processadas", f"{meta.get('paginas_com_texto', 0)}/{meta.get('total_paginas', 0)}")
                
                if meta.get("falhas_estruturais"):
                    st.warning("Falhas de layout detetadas:")
                    for f in meta["falhas_estruturais"]:
                        st.write(f"- {f}")
                else:
                    st.success("✅ Estrutura de leitura limpa. O documento é facilmente legível por leitores ATS.")
else:
    st.title("🚀 ResumePilot AI")
    st.markdown("""
    ### Diagnóstico Determinístico de Currículos para ATS
    
    Este motor simula a triagem de um Applicant Tracking System (ATS) corporativo de última geração:
    
    - **100% Determinístico e Local:** Sem custos de API, sem chamadas de rede e sem alucinações.
    - **Auditoria de Contactos e Cabeçalho:** Validação de e-mail, telefone, links e localização.
    - **Frequência e Impacto de Skills:** Compara a quantidade de menções de cada competência entre a vaga e o currículo.
    - **Kill Questions:** Verificação eliminatória de conselhos de classe e requisitos legais (CREA, CAU, OAB, etc.).
    - **Vigor Executivo vs. Linguagem Passiva:** Identificação de clichês e cálculo de protagonismo.
    
    Envie o seu currículo em PDF na barra lateral, cole a vaga desejada e clique em **Analisar Currículo**.
    """)