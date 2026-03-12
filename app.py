import os
import streamlit as st
from groq import Groq
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer

# cashe - 1 time
@st.cache_resource
def load_clients():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("rag-chatbot")
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, groq_client, embedding_model

index, groq_client, embedding_model = load_clients()

# RAG functions
def get_embeddings(texts):
    return embedding_model.encode(texts, show_progress_bar=False).tolist()

def retrieve_context(query, top_k=5, score_threshold=0.3):
    query_vector = get_embeddings([query])[0]
    results = index.query(vector=query_vector, top_k=top_k, include_metadata=True)
    matches = results["matches"] if isinstance(results, dict) else results.matches
    retrieved_docs = []
    for match in matches:
        metadata = match["metadata"] if isinstance(match, dict) else match.metadata
        score = match["score"] if isinstance(match, dict) else match.score
        if score < score_threshold:
            continue
        retrieved_docs.append({
            "score": score,
            "text": metadata.get("text", ""),
            "doc_type": metadata.get("doc_type", "unknown"),
            "item_name": metadata.get("item_name", ""),
            "category_name": metadata.get("category_name", ""),
            "date": metadata.get("date", ""),
            "item_code": metadata.get("item_code", "")
        })
    return sorted(retrieved_docs, key=lambda x: x["score"], reverse=True)

def create_rag_prompt(query, contexts):
    context_text = "\n\n".join([
        f"[Source: {ctx['doc_type']} | Score: {ctx['score']:.2f}]\n{ctx['text']}"
        for ctx in contexts
    ])
    return f"""You help with veggie sales data for a supermarket.
Answer the user's question based ONLY on the provided context below.
If you can't find the answer, say "I don't have enough information to answer that."

Context:
{context_text}

Question: {query}

Answer:"""

def generate_answer(query, contexts):
    if not contexts:
        return "I could not find any relevant information."
    prompt = create_rag_prompt(query, contexts)
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": "You are a helpful supermarket sales RAG assistant. Answer only from the retrieved context."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=800
    )
    return response.choices[0].message.content

# Streamlit UI
st.title("Supermarket's Veggie Sales Chatbot")
st.caption("APIs - Groq + Pinecone + Sentence Transformers")

# chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# chat history display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# input
if prompt := st.chat_input("Ask about veggie sales, revenue, margins..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking !"):
            contexts = retrieve_context(prompt)
            answer = generate_answer(prompt, contexts)

            if contexts:
                sources = "\n\n**Sources:**\n" + "\n".join([
                    f"- `{doc['doc_type']}` (score: {doc['score']:.2f})" +
                    (f" — {doc['date']}" if doc.get("date") else "")
                    for doc in contexts[:3]
                ])
                full_response = answer + sources
            else:
                full_response = answer

        st.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
