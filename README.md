# ResumePilot AI

Sistema inteligente de análise de currículos utilizando IA do Google Gemini.

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

- ✅ **Análise ATS**: Verifica se o currículo é compatível com sistemas de tracking
- ✅ **Palavras-chave**: Identifica termos críticos ausentes da vaga
- ✅ **Metodologia XYZ**: Sugere reescritas baseadas em resultados mensuráveis
- ✅ **Filtro de adjetivos**: Detecta autoavaliações subjetivas sem evidência
- ✅ **Score de prontidão**: Avaliação quantitativa da adequação à vaga

## 📝 Estrutura do Projeto

```
Recrutamento/
├── app.py              # Aplicação principal Streamlit
├── requirements.txt    # Dependências Python
└── README.md          # Este arquivo
```

## 🔒 Segurança

A API Key é inserida apenas na interface do aplicativo e não é armazenada. Recomenda-se usar variáveis de ambiente em produção:
```bash
set GEMINI_API_KEY=sua_chave_aqui
```

E modificar o código para ler a variável de ambiente em vez do input de texto.
