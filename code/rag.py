import pandas as pd
import os
from langchain.llms import HuggingFacePipeline
from langchain.embeddings import SentenceTransformerEmbeddings
from transformers import pipeline
from langchain.vectorstores import Chroma
from langchain.document_loaders import PyMuPDFLoader
from langchain.chains import RetrievalQA
from tqdm import tqdm
from langchain.prompts import PromptTemplate
import time

llm_model = "meta-llama/Llama-3.2-3B-Instruct"
file_path = 'test.xlsx'
embedding_model = 'all-MiniLM-L6-v2'

embedding_model = SentenceTransformerEmbeddings(model_name=embedding_model)
llm_pipeline  = pipeline("text-generation", model=llm_model,max_length = 4096,max_new_tokens = 200, device=0, token=os.getenv('HF_TOKEN'))
test = pd.read_excel(file_path)

pdf_directory = "/kaggle/input/project"
pdf_names = os.listdir(pdf_directory)
documents = []
total_token_count = 0
for pdf_name in tqdm(pdf_names, desc="Loading PDFs"):
    token_count = 0
    pdf_path = os.path.join(pdf_directory, pdf_name)
    loader = PyMuPDFLoader(pdf_path)
    loaded_documents = loader.load()
    for document in loaded_documents:
        text = document.page_content
        documents.append(document)
        token_count+=len(llm_pipeline.tokenizer(text)["input_ids"])
    print(f"Token at {pdf_name}: {token_count}")
    total_token_count+=token_count
print(f"Token at all PDF: {total_token_count}")

vector_store = Chroma.from_documents(documents, embedding_model)
retriever=vector_store.as_retriever()
prompt_template ="""
    使用所給的上下文來回答問題。
    使用不超過三句來回答，並且用繁體中文。
    Context: {context}
    Question: {question}
    Answer:
"""
prompt = PromptTemplate.from_template(prompt_template)
qa = RetrievalQA.from_chain_type(
    retriever=retriever,
    llm=HuggingFacePipeline(pipeline=llm_pipeline),
    chain_type="stuff",
    chain_type_kwargs = {"prompt": prompt}
)

def ask_question_with_rag(prompt):
    start_time = time.time()
    response = llm_pipeline(prompt+'Answer: ')
    answer = response[0]['generated_text'].split('Answer: ')[1].strip()
    input_token = len(llm_pipeline.tokenizer(prompt)["input_ids"])
    output_token = len(llm_pipeline.tokenizer(answer)["input_ids"])
    num_tokens = len(input_token)
    t = time.time() - start_time
    return input_token, output_token, answer, t
first_tokens = []
final_tokens = []
first_prompts = []
final_prompts = []
first_answers = []
final_answers = []
attempts = []
first_times = []
final_times = []
for query in tqdm(test['query']):
    first_token, final_token, first_prompt, final_prompt, first_answer, final_answer, attempt, first_time, final_time = \
    ask_question_with_rag(query, retry = True)
    first_tokens.append(first_token)
    final_tokens.append(final_token)
    first_prompts.append(first_prompt)
    final_prompts.append(final_prompt)
    first_answers.append(first_answer)
    final_answers.append(final_answer)
    attempts.append(attempt)
    first_times.append(first_time)
    final_times.append(final_time)
