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
model_choice = input("Please enter LLM (gemma / llama/ yi): ")
model, tokenizer = models(model_choice)

split_documents = []
from langchain.schema import Document
from langchain_community.vectorstores import Chroma
pdf_directory = "datasets"
pdf_names = [pdf_name for pdf_name in os.listdir(pdf_directory) if pdf_name.endswith('.pdf')]
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


def ircot_prompt(question: str, context: str, retrieved_info: str) -> str:
    input_text = f"""
    Analyze the question and retrieved information step-by-step. Use the following format and traditional chinese:

    Question: {question}
    Context: {context}
    Retrieved Information: {retrieved_info}

    Reasoning:
    1. Start by addressing the question directly.
    2. Reference key details from the retrieved information.
    3. Connect the dots between the question and retrieved facts.
    4. If confident, conclude with "The answer is: [answer]".
    """
    return input_text
def generate_reasoning(retrieved_context, context, question):
    prompt = ircot_prompt(retrieved_context, context, question)
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = inputs.to('cuda')
    input_token = len(tokenizer(prompt)["input_ids"])
    outputs = model.generate(**inputs,max_length=input_token+200)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def generate_reasoning_llama(retrieved_context, context, question):
    prompt = ircot_prompt(retrieved_context, context, question)
    inputs = tokenizer(
            text = prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(output[0])
    return response

def ircot(question):
    max_iter = 3
    context = ''
    history = {
        "queries": [],
        "retrieved_docs": [],
        "reasoning": [],
    }

    for iteration in range(1, max_iter + 1):

        # Step 1: Generate search query
        if iteration == 1:
            query = question  # Start with the original question
        else:
            query = f"{question} based on {history['reasoning'][-1]}"  # Refine query based on previous reasoning
        history["queries"].append(query)

        # Step 2: Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        history["retrieved_docs"].append(retrieved_docs)
        retrieved_info = "\n".join([doc.page_content for doc in retrieved_docs])  # Combine retrieved documents

        # Step 3: Generate reasoning using the Hugging Face model
        reasoning = generate_reasoning(question, context, retrieved_info)
        history["reasoning"].append(reasoning)

        # Update context for the next iteration
        context += f"\nIteration {iteration} Reasoning: {reasoning}"

        # Step 4: Check if further iterations are needed
        if "answer is" in reasoning.lower():
            final_answer = history["reasoning"][-1]
            model_response = final_answer.split('answer is')[1]
            break
        elif iteration == max_iter:
            final_answer = history["reasoning"][-1]
    return model_response

def ircot_llama(question):
    max_iter = 3
    context = ''
    history = {
        "queries": [],
        "retrieved_docs": [],
        "reasoning": [],
    }

    for iteration in range(1, max_iter + 1):

        # Step 1: Generate search query
        if iteration == 1:
            query = question  # Start with the original question
        else:
            query = f"{question} based on {history['reasoning'][-1]}"  # Refine query based on previous reasoning
        history["queries"].append(query)

        # Step 2: Retrieve relevant documents
        retrieved_docs = retriever.get_relevant_documents(query)
        history["retrieved_docs"].append(retrieved_docs)
        retrieved_info = "\n".join([doc.page_content for doc in retrieved_docs])  # Combine retrieved documents

        # Step 3: Generate reasoning using the Hugging Face model
        reasoning = generate_reasoning_llama(question, context, retrieved_info)
        history["reasoning"].append(reasoning)

        # Update context for the next iteration
        context += f"\nIteration {iteration} Reasoning: {reasoning}"

        # Step 4: Check if further iterations are needed
        if "answer is" in reasoning.lower():
            final_answer = history["reasoning"][-1]
            model_response = final_answer.split('answer is')[1]
            break
        elif iteration == max_iter:
            final_answer = history["reasoning"][-1]
    return model_response


while True:
    user_question = input("Please enter your question (or press Enter to quit): ")
    if user_question == "":
        print("No question entered. Exiting.")
        break
    else:
        if model_choice == "llama":
            response = ircot_llama(user_question)
        else:
            response = response = ircot(user_question)
    print(f"Model Response: {response}")
