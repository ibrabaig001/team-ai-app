import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader

# 🎨 1. App Web Interface Layout
st.set_page_config(page_title="Team AI Custom Portal", layout="wide")
st.title("🧠 Team Custom Knowledge Base")

# 🔑 2. Pulling Keys Securely from Streamlit Advanced Secrets Vault
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_HOST = st.secrets["PINECONE_HOST"]

# Initialize Cloud Connections
try:
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
except Exception as e:
    st.error(f"Configuration Connection Error: {str(e)}")

# 📂 3. Visual Sidebar Document Uploader
with st.sidebar:
    st.header("📁 Document Training Control")
    uploaded_file = st.file_uploader("Upload Team PDFs or Text Files", type=["pdf", "txt"])
    
    if uploaded_file is not None:
        with st.spinner("Processing document data..."):
            raw_text = ""
            if uploaded_file.name.endswith(".pdf"):
                pdf_reader = PdfReader(uploaded_file)
                for page in pdf_reader.pages:
                    raw_text += page.extract_text() or ""
            elif uploaded_file.name.endswith(".txt"):
                raw_text = uploaded_file.read().decode("utf-8")
                
            if raw_text:
                # Turn document text into vectors via OpenAI
                embed_response = ai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=raw_text[:8000] # Safe text limit slice for evaluation testing
                )
                vector = embed_response.data.embedding
                
                # Push vectors to Pinecone cloud filing cabinet
                index.upsert(vectors=[(uploaded_file.name, vector, {"text": raw_text[:2000]})])
                st.success(f"Successfully loaded: {uploaded_file.name}!")

# 💬 4. Live Chat Interface Window
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_query := st.chat_input("Ask anything about your team files..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Analyzing document context..."):
            try:
                query_embed = ai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=user_query
                ).data.embedding
                
                search_results = index.query(vector=query_embed, top_k=1, include_metadata=True)
                
                context = ""
                if search_results.get("matches"):
                    context = search_results["matches"]["metadata"]["text"]
                
                ai_response = ai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are a strict data analysis assistant. Use only this context to answer: {context}. If the information is missing, state 'Not found in files'."},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.1
                ).choices.message.content
                
                st.write(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"Execution Error: {str(e)}")
