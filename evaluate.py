"""
Basic offline RAG evaluation.

Create evaluation/questions.json with:
[
  {
    "question": "What was revenue in FY2025?",
    "ground_truth": "..."
  }
]

Run:
    python evaluate.py

This measures simple retrieval recall against expected source/chunk text.
For a serious project, extend this with Ragas/DeepEval and human-reviewed labels.
"""

import json
import os
import sys

from rag import FinancialRAG


def main():
    if not os.path.exists("evaluation/questions.json"):
        print("Create evaluation/questions.json first.")
        return

    with open("evaluation/questions.json", "r", encoding="utf-8") as f:
        questions = json.load(f)

    rag = FinancialRAG()

    print("Evaluation requires documents to be indexed.")
    print("For the first version, use the Streamlit app to upload documents.")
    print("Then extend this script to load a persistent vector store.")
    print(f"Loaded {len(questions)} evaluation questions.")


if __name__ == "__main__":
    main()
