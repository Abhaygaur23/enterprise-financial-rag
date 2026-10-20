# Enterprise Financial Intelligence RAG

A resume-ready RAG application for financial/enterprise document research.

## Features

- PDF/DOCX/TXT/CSV ingestion
- Page-aware metadata
- Overlapping chunking
- OpenAI embeddings
- Hybrid semantic + BM25 retrieval
- Reciprocal-style weighted fusion
- Lightweight reranking
- Query classification
- Grounded LLM generation
- Inline source citations
- Retrieval diagnostics
- Streamlit UI
- Evaluation scaffold

## Local setup

### 1. Install Python

Use Python 3.11 or 3.12.

### 2. Create environment

Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation:

```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 3. Install packages

```powershell
pip install -r requirements.txt
```

### 4. Add your API key

Create:

`.streamlit/secrets.toml`

```toml
OPENAI_API_KEY = "your-key"
OPENAI_CHAT_MODEL = "gpt-4o-mini"
```

Never commit this file.

### 5. Run

```powershell
streamlit run app.py
```

## Suggested demo

Upload two annual reports from the same company or two companies.

Ask:

- What was revenue in the latest year?
- Why did operating margin change?
- Compare revenue growth between the two companies.
- What risks are mentioned across both reports?
- Which document supports this claim?

## Deployment

Push this repository to GitHub and deploy it with Streamlit Community Cloud.

Add the same values from `secrets.toml` in the deployment Secrets field.
