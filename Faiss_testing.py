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

vectors = None
index = None
model = SentenceTransformer('sentence-t5-base')

History_human = []
Ai_history = []

vector_store_file = "faiss_index.index"
chunks_file = "chunks.npy"


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
    global vectors, index
    chunks = Knowledge_base()
    vectors = model.encode(chunks, convert_to_numpy=True)
    faiss.normalize_L2(vectors)

    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)

    faiss.write_index(index, vector_store_file)
    np.save(chunks_file, np.array(chunks))

    return index, chunks


def load_index():
    global index
    index = faiss.read_index(vector_store_file)
    return index


def load_chunks():
    return np.load(chunks_file, allow_pickle=True).tolist()


def Question_embedding():
    question_embedding = model.encode([question], convert_to_numpy=True)
    faiss.normalize_L2(question_embedding)
    return question_embedding


def find_similarity():
    if not os.path.exists(vector_store_file) or not os.path.exists(chunks_file):
        print("Creating and storing new vector index...")
        index, chunks = Doc_embedding()
    else:
        print("Loading existing vector index...")
        index = load_index()
        chunks = load_chunks()

    q_embedding = Question_embedding()
    D, I = index.search(q_embedding, k=5)

    similarity_threshold = 0.5
    useful_chunks = [chunks[i] for i, score in zip(I[0], D[0]) if 1 - score >= similarity_threshold]
    return useful_chunks


def runRag():
    global Doc_url, question
    Doc_url = "https://benefits.adobe.com/document/275"
    question = input("What do you want to know?\n")

    while question.lower() != "no":
        context = " ".join(find_similarity())
        History_human.append(question)

        prompt = f"""Answer the question based on the context, human chat and AI response. \
        if the questions is irrelevant to the context just say \
        \"The information you're looking for is Not in my knowledge base\" and \
        if the question is relevant to context but you don't have specific answer, \
        just say \"I'm not entirely certain, so I wouldn't want to give you the wrong answer.\" \
        also remember the human chat {History_human} and the AI response {Ai_history} as well\n\nContext: {context}\nQuestion: {question}"""

        genai.configure(api_key="YOUR_API_KEY")
        os.environ['GOOGLE_API_KEY'] = 'YOUR_API_KEY'
        llm = ChatGoogleGenerativeAI(model="YOUR_MODEL_NAME", temperature=0.9)
        response = llm.invoke(prompt)
        Ai_history.append(response)
        print(response.content)

        question = input("What do you want to know?\n")

runRag()