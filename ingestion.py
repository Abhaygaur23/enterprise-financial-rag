import io
import re
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from docx import Document


def clean_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def extract_pdf(file_bytes, filename):
    reader = PdfReader(io.BytesIO(file_bytes))
    docs = []

    for page_no, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            docs.append({
                "text": text,
                "source": filename,
                "location": f"page {page_no}",
            })
    return docs


def extract_docx(file_bytes, filename):
    doc = Document(io.BytesIO(file_bytes))
    text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    text = clean_text(text)

    return [{
        "text": text,
        "source": filename,
        "location": "document",
    }] if text else []


def extract_txt(file_bytes, filename):
    text = clean_text(file_bytes.decode("utf-8", errors="ignore"))
    return [{
        "text": text,
        "source": filename,
        "location": "text file",
    }] if text else []


def extract_csv(file_bytes, filename):
    df = pd.read_csv(io.BytesIO(file_bytes))
    text = df.to_csv(index=False)
    text = clean_text(text)

    return [{
        "text": text,
        "source": filename,
        "location": "CSV",
    }] if text else []


def extract_documents(uploaded_files):
    all_docs = []

    for uploaded in uploaded_files:
        data = uploaded.getvalue()
        suffix = Path(uploaded.name).suffix.lower()

        if suffix == ".pdf":
            all_docs.extend(extract_pdf(data, uploaded.name))
        elif suffix == ".docx":
            all_docs.extend(extract_docx(data, uploaded.name))
        elif suffix == ".txt":
            all_docs.extend(extract_txt(data, uploaded.name))
        elif suffix == ".csv":
            all_docs.extend(extract_csv(data, uploaded.name))

    return all_docs


def chunk_documents(documents, chunk_size=1200, overlap=200):
    chunks = []

    for doc in documents:
        words = doc["text"].split()
        start = 0
        chunk_no = 0

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_text = " ".join(words[start:end]).strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": doc["source"],
                    "location": doc["location"],
                    "chunk_id": f"{doc['source']}-{doc['location']}-{chunk_no}",
                })

            if end >= len(words):
                break

            start = end - overlap
            chunk_no += 1

    return chunks
