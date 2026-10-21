# ResumePilot

Sistema de análise de currículos 100% local, sem dependência de APIs ou chaves externas.

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
pip install streamlit PyMuPDF
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

- ✅ **Análise ATS Local**: Verifica se o currículo é compatível com sistemas de tracking usando PyMuPDF
- ✅ **Palavras-chave**: Identifica termos críticos ausentes da vaga através de análise léxica
- ✅ **Metodologia XYZ**: Sugere reescritas baseadas em resultados mensuráveis
- ✅ **Filtro de adjetivos**: Detecta autoavaliações subjetivas sem evidência (proativo, dinâmico, etc.)
- ✅ **Score de prontidão**: Avaliação quantitativa da adequação à vaga (0-10)
- ✅ **Detecção de métricas**: Identifica números, porcentagens e valores monetários no currículo
- ✅ **100% Gratuito**: Sem necessidade de chaves de API ou serviços externos

## 📝 Estrutura do Projeto

```
Recrutamento/
├── app.py              # Aplicação principal Streamlit
├── requirements.txt    # Dependências Python
└── README.md          # Este arquivo
```

## � Como Funciona

O ResumePilot utiliza lógica pura em Python para analisar currículos:

1. **Extração de Texto**: Usa PyMuPDF para extrair texto de PDFs e avaliar complexidade estrutural
2. **Análise de Palavras-chave**: Extrai termos relevantes da vaga e verifica presença no currículo
3. **Detecção de Adjetivos Esponjosos**: Busca clichês comuns em português (proativo, dinâmico, dedicado, etc.)
4. **Detecção de Métricas**: Usa regex para encontrar porcentagens, valores monetários e números
5. **Cálculo de Score**: Combina todos os fatores para gerar uma nota de prontidão (0-10)

## 🔒 Privacidade

Todas as análises são feitas localmente no seu navegador. Nenhum dado é enviado para servidores externos.
