import streamlit as st
from rag import FinancialRAG
from ingestion import extract_documents

st.set_page_config(
    page_title="Enterprise Financial RAG",
    page_icon="📊",
    layout="wide",
)

@st.cache_resource
def get_rag():
    return FinancialRAG()

rag = get_rag()

st.title("📊 Enterprise Financial Intelligence RAG")
st.caption("Hybrid retrieval + semantic reranking + grounded answers + citations")

with st.sidebar:
    st.header("1. Add documents")
    uploaded = st.file_uploader(
        "Upload PDF, DOCX, TXT or CSV files",
        type=["pdf", "docx", "txt", "csv"],
        accept_multiple_files=True,
    )

    if st.button("Index documents", type="primary", use_container_width=True):
        if not uploaded:
            st.warning("Upload at least one document.")
        else:
            with st.spinner("Reading, chunking and indexing documents..."):
                raw_docs = extract_documents(uploaded)
                result = rag.index_documents(raw_docs)
            st.success(f"Indexed {result['chunks']} chunks from {result['documents']} document(s).")

    st.divider()
    st.header("2. Retrieval settings")
    top_k = st.slider("Final context chunks", 3, 12, 6)
   

    st.divider()
    st.header("Knowledge base")
    st.metric("Documents", len(rag.documents))
    st.metric("Chunks", len(rag.chunks))

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

query = st.chat_input(
    "Ask: Compare revenue growth across the uploaded annual reports..."
)

if query:
    if not rag.chunks:
        st.warning("Please upload and index documents first.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving evidence and generating a grounded answer..."):
            try:
                answer = rag.answer(query, top_k=top_k)
                st.markdown(answer["answer"])

                with st.expander("🔎 Retrieved evidence"):
                    for i, item in enumerate(answer["sources"], 1):
                        st.markdown(
                            f"**[{i}] {item['source']} — {item['location']}**"
                        )
                        st.caption(item["text"])

                with st.expander("🧪 Retrieval diagnostics"):
                    st.json({
                        "query_type": answer["query_type"],
                        "retrieved_candidates": answer["candidate_count"],
                        "final_context": len(answer["sources"]),
                    })
            except Exception as e:
                st.error(f"Error: {e}")

    st.session_state.messages.append({"role": "assistant", "content": answer["answer"]})
