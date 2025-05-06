import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_google_genai import GoogleGenerativeAI
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import google.generativeai as genai
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter

load_dotenv()
os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))






def get_pdf_text(pdf_docs):
    text=""
    for pdf in pdf_docs:
        pdf_reader= PdfReader(pdf)
        for page in pdf_reader.pages:
            text+= page.extract_text()
    return  text



def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks


def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")



# Function to build the QA chain
def get_conversational_chain():
    prompt_template = """
    Answer the question as detailed as possible from the provided context. 
    If the answer is not in the provided context, just say "answer is not available in the context". 
    Do not make up answers.

    Context:
    {context}

    Question: 
    {question}

    Answer:
    """

    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

    # Use ChatGoogleGenerativeAI instead of GoogleGenerativeAI
    llm = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)

    # Load the QA chain with your custom prompt
    chain = load_qa_chain(llm, chain_type="stuff", prompt=prompt)

    return chain

# Function to handle user input
def user_input(user_question):

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    # Embeddings should not include "models/" prefix
    # embeddings = GoogleGenerativeAIEmbeddings(model="embedding-001")

    # Load your FAISS vector store
    # new_db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    new_db = FAISS.from_texts(text_chunks, embeddings)
    new_db.save_local("faiss_index")

    # Search for relevant documents
    docs = new_db.similarity_search(user_question)

    # Get the QA chain
    chain = get_conversational_chain()

    # Run the chain with user question and retrieved docs
    response = chain.invoke({
        "question": user_question,
        "input_documents": docs
    })

    # Print and display response
    print(response)
    st.write("Reply:", response["output_text"])




def main():
    st.set_page_config("Chat with Multiple PDF")
    st.header("Chat with PDF using Google Gemini and Langchain💁")
 
    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_docs = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = get_pdf_text(pdf_docs)
                text_chunks = get_text_chunks(raw_text)
                get_vector_store(text_chunks)
                st.success("Done")



if __name__ == "__main__":
    main()
