import pandas as pd
import os
os.environ['TORCH_USE_CUDA_DSA'] = '1'
import torch
from tqdm import tqdm
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
import re
import time
from llms import models
from langchain.schema import Document

embedding_model = SentenceTransformerEmbeddings(model_name="chuxin-llm/Chuxin-Embedding")
token_counts = {}
pdf_directory = "datasets"
pdf_names = pdf_names = [pdf_name for pdf_name in os.listdir(pdf_directory) if pdf_name.endswith('.pdf')]
documents = []
total_token_count = 0
split_documents = []
for pdf_name in tqdm(pdf_names, desc="Loading PDFs"):
    pdf_path = os.path.join(pdf_directory, pdf_name)
    loader = PyMuPDFLoader(pdf_path)
    loaded_documents = loader.load()
    for document in loaded_documents:
        split_texts = document.page_content.split("\n\n")
        for texts in split_texts:
            split_documents.append(Document(page_content=texts.strip(), metadata=document.metadata))

vector_store = Chroma.from_documents(split_documents, embedding_model)
retriever=vector_store.as_retriever(search_kwargs={"k": 3})

def prompt_rag(context,question):
    return f"""
    使用所給的上下文來回答問題，並且可以使用已有的知識。
    使用不超過三句來回答，並且用繁體中文。
    上下文: {context}
    Question: {question}
    Answer:
    """

def rag_once(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    input_token = len(tokenizer(prompt)["input_ids"])
    outputs = model.generate(**inputs,max_length=input_token+200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    answer = response.split('Answer:\n')[1]
    return answer

def rag_once_llama(prompt):
    inputs = tokenizer(
            text = prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(output[0])
    answer = response.split('Answer:')[1].split(']]>')[0]
    return answer

model_choice = input("Please enter LLM (gemma / llama/ yi): ")
model, tokenizer = models(model_choice)

while True:
    user_question = input("Please enter your question (or press Enter to quit): ")
    if user_question == "":
        print("No question entered. Exiting.")
        break
    else:
        context = retriever.get_relevant_documents(user_question)
        retrieved_content = [item.page_content for item in context]
        prompt = prompt_rag(context, retrieved_content)
        if model_choice == "llama":
            response = rag_once_llama(prompt)
        else:
            response = rag_once(prompt)
    print(f"Model Response: {response}")
