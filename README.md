# ResumePilot

Sistema de análise de currículos baseado em NLP estatístico local usando TF-IDF e PyMuPDF.

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
pip install streamlit PyMuPDF scikit-learn
```

## 📱 Como Usar

### Executar o aplicativo
```bash
streamlit run app.py
```

### Usar o aplicativo
1. Faça upload do seu currículo em PDF na barra lateral
2. Cole a descrição da vaga (Job Description) no campo de texto
3. Clique em "Analisar Currículo"

## 🎯 Funcionalidades

- ✅ **Análise ATS Local (PyMuPDF)**: Avaliação estrutural do PDF - risco de parsing, complexidade de blocos
- ✅ **Análise Estatística (TF-IDF)**: Extração inteligente de palavras-chave com ponderação de relevância
- ✅ **Termos Compostos (N-gramas)**: Captura bigramas como 'gestão financeira' e 'engenharia civil'
- ✅ **Stopwords em Português**: Filtra artigos, preposições e verbos comuns automaticamente
- ✅ **Top 15 Skills**: Extrai as 15 palavras/bigramas mais relevantes da vaga por peso TF-IDF
- ✅ **Score de Prontidão (0-10)**: Nota baseada em 70% proporção de skills + 30% similaridade de cosseno
- ✅ **100% Local**: Rápido, sem API Key, sem custos, sem envio de dados para a nuvem

## 📝 Estrutura do Projeto

```
Recrutamento/
├── app.py              # Aplicação principal Streamlit
├── requirements.txt    # Dependências Python
└── README.md          # Este arquivo
```

## 🔧 Como Funciona

O ResumePilot utiliza técnicas de NLP estatístico:

### 1. Extração de Texto (PyMuPDF)
- Extração de texto do PDF
- Avaliação de risco de parsing (Baixo/Médio/Alto)
- Detecção de falhas estruturais (múltiplas colunas, blocos complexos)

### 2. Análise TF-IDF (scikit-learn)
- Vetorização com stopwords em português
- N-gramas (1, 2) para capturar termos compostos
- Extração das Top 15 features com maiores pesos da vaga
- Verificação de presença no currículo (case-insensitive)
- Cálculo de similaridade de cosseno entre textos

### 3. Cálculo de Score
- 70% do peso: proporção das 15 skills encontradas
- 30% do peso: similaridade global de cosseno
- Score final normalizado para escala 0-10

## 🔒 Privacidade

- Toda a análise é feita 100% localmente
- Nenhum dado é enviado para servidores externos
- Não requer API Key ou serviços de terceiros
- Processamento rápido e privado no seu navegador

