import os
import re
import numpy as np

import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi

from ingestion import chunk_documents


class FinancialRAG:

    def __init__(self):

        # -----------------------------
        # Groq API
        # -----------------------------
        try:
            api_key = st.secrets["GROQ_API_KEY"]
        except Exception:
            api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not found. "
                "Add it to .streamlit/secrets.toml"
            )

        self.client = Groq(api_key=api_key)

        # -----------------------------
        # Groq model
        # -----------------------------
        self.model = "openai/gpt-oss-120b"

        # -----------------------------
        # Local embedding model
        # -----------------------------
        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        # -----------------------------
        # RAG storage
        # -----------------------------
        self.documents = []
        self.chunks = []
        self.embeddings = None
        self.bm25 = None

    # =========================================================
    # EMBEDDINGS
    # =========================================================

    def _embed(self, texts):

        embeddings = self.embedding_model.encode(
            texts,
            batch_size=32,
            show_progress_bar=False,
            normalize_embeddings=True
        )

        return np.asarray(
            embeddings,
            dtype=np.float32
        )

    # =========================================================
    # TOKENIZATION
    # =========================================================

    def _tokenize(self, text):

        return re.findall(
            r"\b\w+\b",
            text.lower()
        )

    # =========================================================
    # INDEX DOCUMENTS
    # =========================================================

    def index_documents(self, documents):

        self.documents = documents

        # Create chunks
        self.chunks = chunk_documents(
            documents,
            chunk_size=1200,
            overlap=200
        )

        if not self.chunks:
            raise ValueError(
                "No text chunks were created from the documents."
            )

        # Extract chunk text
        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        # Local embeddings
        self.embeddings = self._embed(texts)

        # BM25 index
        tokenized_corpus = [
            self._tokenize(text)
            for text in texts
        ]

        self.bm25 = BM25Okapi(
            tokenized_corpus
        )

        return {
            "documents": len(documents),
            "chunks": len(self.chunks)
        }

    # =========================================================
    # SEMANTIC SEARCH
    # =========================================================

    def _semantic_search(
        self,
        query,
        top_k=10
    ):

        query_embedding = self._embed(
            [query]
        )[0]

        scores = np.dot(
            self.embeddings,
            query_embedding
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for idx in top_indices:

            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "location": self.chunks[idx]["location"]
            })

        return results

    # =========================================================
    # BM25 SEARCH
    # =========================================================

    def _bm25_search(
        self,
        query,
        top_k=10
    ):

        query_tokens = self._tokenize(
            query
        )

        scores = self.bm25.get_scores(
            query_tokens
        )

        top_indices = np.argsort(
            scores
        )[::-1][:top_k]

        results = []

        for idx in top_indices:

            results.append({
                "index": int(idx),
                "score": float(scores[idx]),
                "text": self.chunks[idx]["text"],
                "source": self.chunks[idx]["source"],
                "location": self.chunks[idx]["location"]
            })

        return results

    # =========================================================
    # HYBRID RETRIEVAL
    # =========================================================

    def retrieve(
        self,
        query,
        top_k=6
    ):

        semantic_results = self._semantic_search(
            query,
            top_k=10
        )

        bm25_results = self._bm25_search(
            query,
            top_k=10
        )

        combined = {}

        # Semantic results
        for rank, result in enumerate(
            semantic_results
        ):

            idx = result["index"]

            combined.setdefault(
                idx,
                {
                    "index": idx,
                    "text": result["text"],
                    "source": result["source"],
                    "location": result["location"],
                    "score": 0.0
                }
            )

            combined[idx]["score"] += (
                1.0 / (rank + 1)
            )

        # BM25 results
        for rank, result in enumerate(
            bm25_results
        ):

            idx = result["index"]

            combined.setdefault(
                idx,
                {
                    "index": idx,
                    "text": result["text"],
                    "source": result["source"],
                    "location": result["location"],
                    "score": 0.0
                }
            )

            combined[idx]["score"] += (
                1.0 / (rank + 1)
            )

        # Sort by combined score
        results = sorted(
            combined.values(),
            key=lambda x: x["score"],
            reverse=True
        )

        return results[:top_k]

    # =========================================================
    # QUERY CLASSIFICATION
    # =========================================================

    def classify_query(self, query):

        query_lower = query.lower()

        if any(
            word in query_lower
            for word in [
                "compare",
                "difference",
                "versus",
                "vs",
                "higher",
                "lower"
            ]
        ):
            return "comparison"

        if any(
            word in query_lower
            for word in [
                "why",
                "explain",
                "reason",
                "impact"
            ]
        ):
            return "reasoning"

        if any(
            word in query_lower
            for word in [
                "summarize",
                "summary",
                "overview"
            ]
        ):
            return "summary"

        return "fact"

    # =========================================================
    # GROQ GENERATION
    # =========================================================

    def _generate_answer(
        self,
        query,
        sources,
        query_type
    ):

        context_parts = []

        for i, source in enumerate(
            sources,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE [{i}]
Document: {source["source"]}
Location: {source["location"]}

{source["text"]}
"""
            )

        context = "\n".join(
            context_parts
        )

        system_prompt = """
You are an enterprise financial research assistant.

Answer questions using ONLY the supplied document evidence.

Rules:

1. Do not invent facts.
2. Do not use outside knowledge.
3. If the evidence is insufficient, say so.
4. Cite every important factual claim using [1], [2], etc.
5. For financial comparisons, clearly show the values.
6. If calculations are possible from the provided evidence,
   perform them carefully.
7. Explain your reasoning clearly but concisely.
8. Never fabricate a citation.
"""

        user_prompt = f"""
Query type: {query_type}

User question:
{query}

Document evidence:
{context}

Answer the user's question using the evidence above.

Include source citations such as [1], [2], [3].
"""

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[
                {
                    "role": "system",
                    "content": system_prompt
                },
                {
                    "role": "user",
                    "content": user_prompt
                }
            ],

            temperature=0.1,

            max_tokens=1500
        )

        return response.choices[0].message.content

    # =========================================================
    # PUBLIC ANSWER METHOD
    # =========================================================

    def answer(
        self,
        query,
        top_k=6
    ):

        if not self.chunks:
            raise ValueError(
                "No documents indexed."
            )

        query_type = self.classify_query(
            query
        )

        sources = self.retrieve(
            query,
            top_k=top_k
        )

        answer = self._generate_answer(
            query,
            sources,
            query_type
        )

        return {
            "answer": answer,
            "sources": sources,
            "query_type": query_type,
            "candidate_count": len(sources)
        }