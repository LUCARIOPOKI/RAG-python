# RAG-python
A Retrieval-Augmented Generation system using python

## Chunking & Embedding:
1) Getting a pdf from internet
2) Chunking it using the sent_tokenize and store it in a numpy file
3) Create Embeddings using Sentence transformer
4) Normalize the vector using FAISS normalize method
5) Write the vectors in a FAISS index file

## Finding similarity
1) Load the index from the index file
2) Load the chunks from the chunks file
3) Get the question from the user and embedd it as well and normalize it
4) Compare and return the useful chunks

## Model
1) Here we use Gemini model
2) While the question is not "no" the model will run again and again creating a chat feel
3) We are maintaining the chat history in a list and passing it to the prompt 
