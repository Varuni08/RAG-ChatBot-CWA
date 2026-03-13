import os
import streamlit as st
from groq import Groq
from pinecone import Pinecone
import pandas as pd
import matplotlib.pyplot as plt
import re
from sentence_transformers import SentenceTransformer

    
# cache - 1 time
@st.cache_resource
def load_clients():
    pc = Pinecone(api_key=st.secrets["PINECONE_API_KEY"])
    index = pc.Index("rag-chatbot")
    groq_client = Groq(api_key=st.secrets["GROQ_API_KEY"])
    embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
    return index, groq_client, embedding_model


index, groq_client, embedding_model = load_clients()
df = load_data()

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

# graph plotting
def show_context_graphs(contexts):
    if not contexts:
        return

    df = pd.DataFrame(contexts)

    if df.empty:
        return

    st.subheader("Retrieved Data Insights")

    # scores by item
    if "item_name" in df.columns and "score" in df.columns:
        plot_df = df[df["item_name"].astype(str).str.strip() != ""][["item_name", "score"]].head(10)

        if not plot_df.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(plot_df["item_name"], plot_df["score"])
            ax.set_title("Top Retrieved Items by Similarity Score")
            ax.set_xlabel("Item Name")
            ax.set_ylabel("Score")
            plt.xticks(rotation=45, ha="right")
            st.pyplot(fig)

    # doc type count
    if "doc_type" in df.columns:
        type_counts = df["doc_type"].value_counts()

        if not type_counts.empty:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.bar(type_counts.index, type_counts.values)
            ax.set_title("Retrieved Sources by Document Type")
            ax.set_xlabel("Document Type")
            ax.set_ylabel("Count")
            plt.xticks(rotation=30, ha="right")
            st.pyplot(fig)

    # Show raw retrieved table
    st.dataframe(df)

def is_graph_request(prompt):
    prompt = prompt.lower()
    graph_words = ["graph", "chart", "plot", "visualize", "visualisation", "visualization", "trend"]
    return any(word in prompt for word in graph_words)

# year wise sales
def show_yearwise_sales_graph(df):
    df = df.copy()

    # make sure date column is datetime
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

    # create sales if not already present
    if "Sales" not in df.columns:
        df["Sales"] = df["Unit Selling Price"] * df["Quantity Sold"]

    df["Year"] = df["Date"].dt.year
    year_sales = df.groupby("Year", as_index=False)["Sales"].sum()

    st.subheader("Year-wise Sales Graph")
    st.dataframe(year_sales)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(year_sales["Year"].astype(str), year_sales["Sales"])
    ax.set_title("Year-wise Sales")
    ax.set_xlabel("Year")
    ax.set_ylabel("Sales")
    st.pyplot(fig)


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

    contexts = []  # initialize so it's always defined

    with st.chat_message("assistant"):
        with st.spinner("Thinking !"):
            # graph request handling
            if is_graph_request(prompt) and "year" in prompt.lower() and "sales" in prompt.lower():
                st.markdown("Here is the year-wise sales graph:")
                show_yearwise_sales_graph(df)
                full_response = "Displayed year-wise sales graph."

            else:
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
        st.divider()
        show_context_graphs(contexts)
    st.session_state.messages.append({"role": "assistant", "content": full_response})
