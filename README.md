# ResumePilot

Sistema híbrido de análise de currículos combinando análise estrutural local (PyMuPDF) com análise semântica avançada (Google Gemini AI).

## 🚀 Instalação

### 1. Criar ambiente virtual (recomendado)
```bash
python -m venv venv
```

### 2. Ativar ambiente virtual

**Windows:**
```bash
venv\Scripts\activate
```

**Linux/Mac:**
```bash
source venv/bin/activate
```

### 3. Instalar dependências
```bash
pip install -r requirements.txt
```

Ou instalar manualmente:
```bash
pip install streamlit PyMuPDF google-generativeai
```

## 📱 Como Usar

### 1. Obter API Key do Google Gemini
- Acesse [Google AI Studio](https://makersuite.google.com/app/apikey)
- Crie uma API Key gratuita
- Copie a chave para uso no aplicativo

### 2. Executar o aplicativo
```bash
streamlit run app.py
```

### 3. Usar o aplicativo
1. Faça upload do seu currículo em PDF na barra lateral
2. Cole a descrição da vaga (Job Description) no campo de texto
3. Insira sua API Key do Google Gemini
4. Clique em "Analisar Currículo"

## 🎯 Funcionalidades

- ✅ **Análise ATS Local (PyMuPDF)**: Avaliação estrutural do PDF sem IA - risco de parsing, complexidade de blocos
- ✅ **Análise Semântica (Gemini AI)**: Compreensão contextual de Hard Skills e área de atuação
- ✅ **Extração Inteligente de Hard Skills**: Identifica apenas tecnologias, ferramentas e metodologias reais (ignora verbos e pronomes)
- ✅ **Compatibilidade Semântica**: Avalia contexto real da área (ex: Gestão de TI vs Engenharia Civil)
- ✅ **Metodologia Google XYZ**: Reescrita de frases REAIS do currículo com métricas (não texto inventado)
- ✅ **Filtro de Adjetivos Esponjosos**: Detecta clichês sem evidência factual (proativo, dinâmico, etc.)
- ✅ **Score Contextual (0-10)**: Nota baseada em compatibilidade semântica real, não apenas coincidência de palavras

## 📝 Estrutura do Projeto

```
Recrutamento/
├── app.py              # Aplicação principal Streamlit
├── requirements.txt    # Dependências Python
└── README.md          # Este arquivo
```

## 🔧 Arquitetura Híbrida

O ResumePilot combina duas abordagens:

### 1. Análise Local (PyMuPDF)
- Extração de texto do PDF
- Avaliação de risco de parsing (Baixo/Médio/Alto)
- Detecção de falhas estruturais (múltiplas colunas, blocos complexos)
- Todas essas análises são feitas localmente, sem enviar dados para a nuvem

### 2. Análise Semântica (Google Gemini)
- Extração de Hard Skills reais da vaga (tecnologias, metodologias)
- Análise de compatibilidade semântica de contexto
- Detecção de adjetivos esponjosos
- Extração de frase real do currículo para reescrita XYZ
- Cálculo de score considerando contexto semântico

## 🔒 Privacidade

- A análise estrutural do PDF é feita 100% localmente
- Apenas o texto extraído é enviado ao Gemini para análise semântica
- A API Key é inserida na interface e não é armazenada
- Recomenda-se usar variáveis de ambiente em produção para maior segurança
