import streamlit as st
import faiss
import requests
import nltk
import fitz
from nltk.tokenize import sent_tokenize
import io
from sentence_transformers import SentenceTransformer
import numpy as np
import google.generativeai as genai
import os
from langchain_google_genai import ChatGoogleGenerativeAI

# nltk.download("punkt")
model = SentenceTransformer('sentence-t5-base')
vector_store_file = "faiss_index.index"
chunks_file = "chunks.npy"

# Set page config
st.set_page_config(page_title="RAG Assistant", layout="wide")

# Initialize session state
if 'history_human' not in st.session_state:
    st.session_state.history_human = []
if 'history_ai' not in st.session_state:
    st.session_state.history_ai = []

# Sidebar for doc URL
st.sidebar.title("Document Settings")
Doc_url = st.sidebar.text_input("Enter the doc URL", "https://benefits.adobe.com/document/275")


def Knowledge_base():
    def download_pdf(Doc_url):
        response = requests.get(Doc_url)
        response.raise_for_status()
        return io.BytesIO(response.content)

    def getText(pdf_bytes):
        text = ""
        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
        return text

    def chunk_sentence(text):
        return sent_tokenize(text)

    def GetChunk():
        ByteContent = download_pdf(Doc_url)
        text = getText(ByteContent)
        chunk = chunk_sentence(text)
        return chunk

    return GetChunk()


def Doc_embedding():
    chunks = Knowledge_base()
    vectors = model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(vectors)
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    faiss.write_index(index, vector_store_file)
    np.save(chunks_file, np.array(chunks))
    return index, chunks


def load_index():
    return faiss.read_index(vector_store_file)


def load_chunks():
    return np.load(chunks_file, allow_pickle=True).tolist()


def Question_embedding(question):
    q_embedding = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(q_embedding)
    return q_embedding


def find_similarity(question):
    if not os.path.exists(vector_store_file) or not os.path.exists(chunks_file):
        index, chunks = Doc_embedding()
    else:
        index = load_index()
        chunks = load_chunks()

    q_embedding = Question_embedding(question)
    D, I = index.search(q_embedding, k=5)

    similarity_threshold = 0.5
    useful_chunks = [chunks[i] for i, score in zip(I[0], D[0]) if 1 - score >= similarity_threshold]
    return useful_chunks


def get_response(prompt):
    genai.configure(api_key="YOUR_GOOGLE_API_KEY")
    os.environ['GOOGLE_API_KEY'] = 'YOUR_GOOGLE_API_KEY'
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro-exp-03-25", temperature=0.9)
    return llm.invoke(prompt).content


st.title("📄 Chat with Your Document (RAG + FAISS)")
st.markdown("Chat with the content of a PDF document using Retrieval-Augmented Generation.")

question = st.chat_input("Ask a question about the document...")

if question:
    st.session_state.history_human.append(question)

    context = " ".join(find_similarity(question))
    prompt = f"""
    You are an Adobe Leave and Holiday policy Bot, make your response more human like,
    keep the response short and concise. 
    Answer the question based on the context, human chat and AI response.
    If the question is irrelevant to the context just say:
    "The information you're looking for is Not in my knowledge base."
    If the question is relevant but the answer is unknown, say:
    "I'm not entirely certain, so I wouldn't want to give you the wrong answer."
    Remember the human chat {st.session_state.history_human} and 
    the AI responses {st.session_state.history_ai} and use this data aswell.
    Context: {context}
    Question: {question}
    """

    response = get_response(prompt)
    st.session_state.history_ai.append(response)

for human, ai in zip((st.session_state.history_human),(st.session_state.history_ai)):
    with st.chat_message("user"):
        st.markdown(human)
    with st.chat_message("assistant"):
        st.markdown(ai)
