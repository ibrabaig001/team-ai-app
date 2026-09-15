import streamlit as st
from openai import OpenAI
from pinecone import Pinecone
from pypdf import PdfReader

# 🎨 1. App Interface Configuration
st.set_page_config(page_title="Personalized AI Portal", layout="wide")
st.title("🗂️ Private Document Intelligence Center")

# 🔑 2. Secure Cloud Credentials
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_HOST = st.secrets["PINECONE_HOST"]

try:
    ai_client = OpenAI(api_key=OPENAI_API_KEY)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index(host=PINECONE_HOST)
except Exception as e:
    st.error(f"Cloud Connection Error: {str(e)}")

# Text segmenting logic
def chunk_text(text, chunk_size=1500, overlap=150):
    return [text[i:i + chunk_size] for i in range(0, len(text), chunk_size - overlap)]

# 📂 3. The "AnythingLLM" Style File Storage System
if "trained_files" not in st.session_state:
    st.session_state.trained_files = []

with st.sidebar:
    st.header("📥 Data Feeder Control Panel")
    uploaded_files = st.file_uploader("Feed documents to your private AI", type=["pdf", "txt"], accept_multiple_files=True)
    
    if st.button("🚀 Process & Embed Files"):
        if uploaded_files:
            for uploaded_file in uploaded_files:
                if uploaded_file.name not in st.session_state.trained_files:
                    with st.spinner(f"Reading and vectorizing: {uploaded_file.name}..."):
                        raw_text = ""
                        if uploaded_file.name.endswith(".pdf"):
                            pdf_reader = PdfReader(uploaded_file)
                            for page in pdf_reader.pages:
                                raw_text += page.extract_text() or ""
                        elif uploaded_file.name.endswith(".txt"):
                            raw_text = uploaded_file.read().decode("utf-8")
                        
                        if raw_text:
                            text_chunks = chunk_text(raw_text)
                            for idx, chunk in enumerate(text_chunks):
                                embed_response = ai_client.embeddings.create(
                                    model="text-embedding-3-small",
                                    input=chunk
                                )
                                vector = embed_response.data[0].embedding
                                unique_id = f"{uploaded_file.name}_chunk_{idx}"
                                index.upsert(vectors=[(unique_id, vector, {"text": chunk, "filename": uploaded_file.name})])
                            
                            st.session_state.trained_files.append(uploaded_file.name)
            st.success("All documents successfully saved and indexed in your cloud!")
        else:
            st.warning("Please drag in at least one file first.")
            
    # Visual list showing what the AI currently knows
    st.write("---")
    st.subheader("📚 Active Knowledge Base")
    if st.session_state.trained_files:
        for file in st.session_state.trained_files:
            st.caption(f"🟢 {file}")
        if st.button("🗑️ Clear Local Tracker"):
            st.session_state.trained_files = []
            st.rerun()
    else:
        st.caption("No data fed yet. Your AI is empty.")

# 💬 4. Live Chat Interface Window
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if user_query := st.chat_input("Ask anything about your loaded files..."):
    with st.chat_message("user"):
        st.write(user_query)
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    with st.chat_message("assistant"):
        with st.spinner("Scanning internal document storage..."):
            try:
                query_embed = ai_client.embeddings.create(
                    model="text-embedding-3-small",
                    input=user_query
                ).data[0].embedding
                
                # Queries your specific files up to 5 comprehensive text segments
                search_results = index.query(vector=query_embed, top_k=5, include_metadata=True)
                
                context_segments = []
                if search_results.get("matches"):
                    for match in search_results["matches"]:
                        if "metadata" in match and "text" in match["metadata"]:
                            context_segments.append(match["metadata"]["text"])
                
                context = "\n---\n".join(context_segments)
                
                ai_response = ai_client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": f"You are a personalized document analysis bot. Answer using only this context: {context}. If the information is completely missing, state 'Not found in files'."},
                        {"role": "user", "content": user_query}
                    ],
                    temperature=0.1
                ).choices[0].message.content
                
                st.write(ai_response)
                st.session_state.messages.append({"role": "assistant", "content": ai_response})
                
            except Exception as e:
                st.error(f"Processing Error: {str(e)}")
