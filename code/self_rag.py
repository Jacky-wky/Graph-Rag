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

from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
judge_prompt = PromptTemplate(
    template="""You are a grader assessing relevance of a retrieved document to a user question. \n
    Here is the retrieved document: \n\n {document} \n\n
    Here is the user question: {question} \n
    If the document contains keywords related to the user question, grade it as relevant. \n
    It does not need to be a stringent test. The goal is to filter out erroneous retrievals. \n
    Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question. \n
    Provide the binary score as a JSON with a single key 'score' and no premable or explanation.""",
    input_variables=["question", "document"],
)
rag_prompt = PromptTemplate(
    template="""You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. \n
    If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise.
    Question: {question}
    Context: {context}
    Answer:""",
    input_variables=["question", "context"],
)
hac_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is grounded in / supported by a set of facts. \n
    Here are the facts:
    \n ------- \n
    {documents}
    \n ------- \n
    Here is the answer: {generation}
    Give a binary score 'yes' or 'no' score to indicate whether the answer is grounded in / supported by a set of facts. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "documents"],
)
rewrite_prompt = PromptTemplate(
    template="""You a question re-writer that converts an input question to a better version that is optimized \n
     for vectorstore retrieval. Look at the initial and formulate an improved question. \n
     Here is the initial question: \n\n {question}.,
    Provide a improved question as a JSON with a single key 'question' and no premable or explanation.""",
    input_variables=["question"],
)
rel_prompt = PromptTemplate(
    template="""You are a grader assessing whether an answer is useful to resolve a question. \n
    Here is the answer:
    \n ------- \n
    {generation}
    \n ------- \n
    Here is the question: {question}
    Give a binary score 'yes' or 'no' to indicate whether the answer is useful to resolve a question. \n
    Provide the binary score as a JSON with a single key 'score' and no preamble or explanation.""",
    input_variables=["generation", "question"],
)

from langchain_core.output_parsers.json import JsonOutputParser
from langchain_core.output_parsers.json import OutputParserException
max_iter = 3

def text_llm(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to('cuda')
    input_token = len(tokenizer(prompt)["input_ids"])
    outputs = model.generate(**inputs,max_length=input_token+50)
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

def vision_llm(prompt):
    inputs = tokenizer(
            text = prompt,
            add_special_tokens=False,
            return_tensors="pt"
        ).to(model.device)
    output = model.generate(**inputs, max_new_tokens=200)
    response = tokenizer.decode(output[0])
    return response

def is_valid_json(response):
    try:
        JsonOutputParser().parse(response)
        return True
    except OutputParserException:
        return False

def judge_document(query, docs):
    relevant_documents = []
    for i in range(len(docs)):
        doc_txt = docs[i].page_content
        prompt = judge_prompt.format(question = query, document = doc_txt)
        for attempt in range(max_iter):
            response = text_llm(prompt)
            if is_valid_json(response):
                break
        parsed_response = JsonOutputParser().parse(response)
        if 'yes' in parsed_response['score']:
            relevant_documents.append(doc_txt)
    return relevant_documents

def generate_response(query, relevant_documents):
    prompt = rag_prompt.format(question = query, context = relevant_documents)
    response = text_llm(prompt)
    result = StrOutputParser().parse(response)
    return result

def judge_hac(response, relevant_documents):
    prompt = hac_prompt.format(generation = response, documents = relevant_documents)
    for _ in range(max_iter):
        response = text_llm(prompt)
        if is_valid_json(response):
            break
    parsed_response = JsonOutputParser().parse(response)
    return parsed_response

def judge_rel(response, query):
    prompt = rel_prompt.format(generation = response, question = query)
    for _ in range(max_iter):
        response = text_llm(prompt)
        if is_valid_json(response):
            break
    parsed_response = JsonOutputParser().parse(response)
    is_rel = parsed_response['score']
    return is_rel
def rewrite_question(query):
    prompt = rewrite_prompt.format(question = query)
    for _ in range(max_iter):
        response = text_llm(prompt)
        if is_valid_json(response):
            break
    parsed_response = JsonOutputParser().parse(response)
    query = parsed_response['question']
    return query

def judge_document_llama(query, docs):
    relevant_documents = []
    for i in range(len(docs)):
        doc_txt = docs[i].page_content
        prompt = judge_prompt.format(question = query, document = doc_txt)
        for attempt in range(max_iter):
            response = vision_llm(prompt)
            if is_valid_json(response):
                break
        parsed_response = JsonOutputParser().parse(response)
        if 'yes' in parsed_response['score']:
            relevant_documents.append(doc_txt)
    return relevant_documents

def generate_response_llama(query, relevant_documents):
    prompt = rag_prompt.format(question = query, context = relevant_documents)
    response = vision_llm(prompt)
    result = StrOutputParser().parse(response)
    return result

def judge_hac_llama(response, relevant_documents):
    prompt = hac_prompt.format(generation = response, documents = relevant_documents)
    for _ in range(max_iter):
        response = vision_llm(prompt)
        if is_valid_json(response):
            break
    parsed_response = JsonOutputParser().parse(response)
    return parsed_response

def judge_rel_llama(response, query):
    prompt = rel_prompt.format(generation = response, question = query)
    for _ in range(max_iter):
        response = vision_llm(prompt)
        if is_valid_json(response):
            break
    parsed_response = JsonOutputParser().parse(response)
    is_rel = parsed_response['score']
    return is_rel
def rewrite_question_llama(query):
    prompt = rewrite_prompt.format(question = query)
    for _ in range(max_iter):
        response = vision_llm(prompt)
        if is_valid_json(response):
            break
    parsed_response = JsonOutputParser().parse(response)
    query = parsed_response['question']
    return query

def self_rag(query):
    while True:
        print('Retrieving files')
        docs = retriever.get_relevant_documents(query)
        relevant_documents = judge_document(query, docs)
        if relevant_documents == []:
            print('No relevant files, we are rewriting the question now')
            print('Rewrite the problem as:')
            query = rewrite_question(query)
            print(query)
        else:
            print('There are relevant documents')
            print('Generating answers')
            response = generate_response(query, relevant_documents)
            print('Answer:', response)
            is_hac = judge_hac(response, relevant_documents)
            if 'yes' in is_hac:
                print('This answer is an hallucination, we need to generate a new answer')
                continue
            else:
                print('This answer is not an hallucination, we now determine if the answer helps solve the problem ')
                is_rel = judge_rel(response, query)
                if 'yes' in is_rel:
                    print('This response helps resolve the query')
                    break
                else:
                    print('This response did not help resolve the query, we are now rewriting the question')
                    query = rewrite_question(query)
                    print('Rewrite the problem as:')
                    print(query)

def self_rag_llama(query):
    while True:
        print('Retrieving files')
        docs = retriever.get_relevant_documents(query)
        relevant_documents = judge_document_llama(query, docs)
        if relevant_documents == []:
            print('No relevant files, we are rewriting the question now')
            print('Rewrite the problem as:')
            query = rewrite_question_llama(query)
            print(query)
        else:
            print('There are relevant documents')
            print('Generating answers')
            response = generate_response_llama(query, relevant_documents)
            print('Answer:', response)
            is_hac = judge_hac_llama(response, relevant_documents)
            if 'yes' in is_hac:
                print('This answer is an hallucination, we need to generate a new answer')
                continue
            else:
                print('This answer is not an hallucination, we now determine if the answer helps solve the problem ')
                is_rel = judge_rel_llama(response, query)
                if 'yes' in is_rel:
                    print('This response helps resolve the query')
                    break
                else:
                    print('This response did not help resolve the query, we are now rewriting the question')
                    query = rewrite_question_llama(query)
                    print('Rewrite the problem as:')
                    print(query)

while True:
    user_question = input("Please enter your question (or press Enter to quit): ")
    if user_question == "":
        print("No question entered. Exiting.")
        break
    else:
        if model_choice == "llama":
            response = self_rag_llama(user_question)
        else:
            response = response = self_rag(user_question)
    print(f"Model Response: {response}")
