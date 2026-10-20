import streamlit as st
from groq import Groq

client = Groq(
    api_key=st.secrets["GROQ_API_KEY"]
)

response = client.chat.completions.create(
    model="openai/gpt-oss-120b",
    messages=[
        {
            "role": "user",
            "content": "Explain RAG in one simple sentence."
        }
    ],
)

st.title("Groq API Test")

st.write(response.choices[0].message.content)