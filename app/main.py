import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
from nltk.tokenize import sent_tokenize
from collections import Counter
import nltk
import json
import os
import base64

# Load environment variables
load_dotenv()
nltk.download('punkt')

# Streamlit Config 
st.set_page_config(page_title="IntelliJect", layout="wide")

# Embed background image and theme
with open(os.path.join("assets", "background.jpg"), "rb") as img:
    bg_data = base64.b64encode(img.read()).decode()

st.markdown(f"""
    <style>
    [data-testid="stAppViewContainer"] {{
        background: url("data:image/png;base64,{bg_data}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }}
    [data-testid="stAppViewContainer"]::before {{
        content: "";
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background-color: rgba(0, 0, 0, 0.75);
        z-index: 0;
    }}
    * {{
        color: #f1e5c0 !important;
        font-family: 'Georgia', serif;
        font-size: 16px;
    }}
    textarea {{
        background-color: rgba(20, 20, 20, 0.7) !important;
        color: #f8f0db !important;
        border: 1px solid #b49c7a !important;
    }}
    button {{
        background-color: #5a4738 !important;
        color: #f3e3c3 !important;
        border-radius: 5px;
    }}
    </style>
""", unsafe_allow_html=True)

# Title 
st.title("📄 IntelliJect - Intelligent PYQ Injection into Notes")
st.markdown("Upload your notes and match them with relevant PYQs from a selected subject.")

# File Paths
subject_folder = os.path.join("app", "subjects")
subject_files = [f for f in os.listdir(subject_folder) if f.endswith('.json')]
subjects = [os.path.splitext(f)[0] for f in subject_files]

# UI 
col1, col2 = st.columns(2)
with col1:
    pdf_file = st.file_uploader("📑 Upload Notes PDF", type=["pdf"])
with col2:
    selected_subject = st.selectbox("📚 Select Subject", subjects)

chunk_size = 10
threshold = 0.93
run_button = st.button("🚀 Inject PYQs")

# Logic 
if run_button:
    if not pdf_file:
        st.error("❌ Please upload a Notes PDF.")
        st.stop()

    with st.spinner("Processing..."):
        try:
            pyq_path = os.path.join(subject_folder, f"{selected_subject}.json")
            with open(pyq_path, "r", encoding="utf-8") as f:
                pyq_data = json.load(f)
        except Exception as e:
            st.error(f"⚠ Failed to load PYQ JSON: {e}")
            st.stop()

        def get_most_common_pyqs(pyq_data, top_k=3):
            question_counts = Counter(q["question"].strip().lower() for q in pyq_data)
            return set(q for q, count in question_counts.items() if count >= top_k)

        common_pyqs = get_most_common_pyqs(pyq_data)

        with open("temp_uploaded.pdf", "wb") as f:
            f.write(pdf_file.read())

        loader = PyPDFLoader("temp_uploaded.pdf")
        pages = loader.load()
        full_text = "\n".join([page.page_content for page in pages])

        def chunk_notes(text, chunk_size=10):
            sentences = sent_tokenize(text)
            return [" ".join(sentences[i:i + chunk_size]) for i in range(0, len(sentences), chunk_size)]

        chunks = chunk_notes(full_text, chunk_size)

        embedding_model = OpenAIEmbeddings()
        llm = ChatOpenAI(model="gpt-4o")

        def infer_subtopic(chunk):
            prompt = f"""
You are a university assistant. Read the note and return the most specific subtopic name.

Note:
\"\"\"{chunk}\"\"\"

Examples: 'Green Chemistry and its Basic Principles', 'Matrix Operations', etc.
Respond only with the subtopic name.
"""
            try:
                return llm.invoke(prompt).content.strip()
            except:
                return "unknown"

        def filter_pyqs_by_subtopic_semantic(subtopic, threshold=0.93):
            if not subtopic or subtopic == "unknown":
                return []
            try:
                query_vec = embedding_model.embed_query(subtopic)
                scored = []
                for q in pyq_data:
                    meta_sub = q.get("sub_topic", "")
                    doc_vec = embedding_model.embed_query(meta_sub)
                    dot = sum(a * b for a, b in zip(query_vec, doc_vec))
                    scored.append((q, dot))
                return sorted([q for q, sim in scored if sim >= threshold], key=lambda x: x['marks'], reverse=True)
            except:
                return []

        def semantic_match_pyqs(pyqs, chunk):
            if not pyqs:
                return []
            try:
                docs = [Document(page_content=q["question"], metadata=q) for q in pyqs]
                vector_store = FAISS.from_documents(docs, embedding=embedding_model)
                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 2})
                return retriever.invoke(chunk)
            except:
                return []

        for i, chunk in enumerate(chunks):
            subtopic = infer_subtopic(chunk)
            filtered = filter_pyqs_by_subtopic_semantic(subtopic, threshold)
            matches = semantic_match_pyqs(filtered, chunk)
            matches = [doc for doc in matches if doc.page_content.strip().lower() not in common_pyqs]

            col1, col2 = st.columns(2)
            with col1:
                st.markdown(f"#### 🧠 {subtopic}")
                st.text_area(label="", value=chunk, height=300, key=f"note_{i}")
            with col2:
                st.markdown("#### 📘 Related PYQs")
                if matches:
                    for doc in matches:
                        st.markdown(f"- {doc.page_content} (Subtopic: {doc.metadata.get('sub_topic', 'N/A')})")
                else:
                    st.write("No unique PYQs found.")

        st.success("✅ Done! Your notes are now enriched with relevant PYQs.")
