__import__('pysqlite3')
import sys
import base64
from pdf2image import convert_from_path
from PIL import Image
import streamlit as st
import tempfile
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from typing import List, Dict
from openai import OpenAI
import chromadb
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from tika import parser
from langchain_community.llms import DeepInfra
import os
from langchain_openai import OpenAIEmbeddings
import chromadb
import re
import PyPDF2
from unstructured.documents.elements import Header, Footer
import string
from streamlit_pdf_viewer import pdf_viewer

# Streamlit app layout with theme

# Configure Streamlit page
st.set_page_config(
    page_title="VŠE AI Study Buddy",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Cache models and embeddings to avoid reloading them
@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings()

@st.cache_resource
def load_vectorstore(_embeddings):
    return Chroma(persist_directory='db', embedding_function=_embeddings)

@st.cache_resource
def load_llm():
    llm = DeepInfra(model_id="mistralai/Mixtral-8x22B-Instruct-v0.1", deepinfra_api_token="hIvZQRN11e1BLIeYghOFCahQYX18uXeY")
    llm.model_kwargs = {
        "temperature": 0.4,
        "repetition_penalty": 1.2,
        "max_new_tokens": 500,
        "top_p": 0.90,
    }
    return llm

@st.cache_resource
def load_prompt():
    return PromptTemplate(
        template="""
        You are a study assistant for students at the University of Economics in Prague. Your task is to answer questions, summarize texts, and assist with learning. Follow these guidelines:
        1. Be polite and answer questions accurately.
        2. Respond in the language in which the question is asked. If the question is in Czech, respond in Czech; if in English, respond in English.
        3. Provide references or sources when possible.
        4. Keep responses concise and to the point.
        """
    )

# Cache parsed PDF content
@st.cache_data
def parse_pdf(file):
    return parser.from_file(file)['content']

# Cache AI-generated responses
@st.cache_data
def generate_response(prompt, history=[]):
    # Logic to generate response (this should include calling the appropriate model and handling the history)
    llm = load_llm()
    prompt_template = load_prompt()
    response = llm.generate(prompt_template.format_prompt(prompt=prompt))
    return response, "example.pdf", history

# Streamlit app layout
st.sidebar.image('logo_fis.jpg', width=150)
st.sidebar.markdown("# VŠE AI Study Buddy")

col1, col2 = st.columns([2, 1])
with col2:
    st.markdown("<img src='path/to/logo.png' width='150' style='float:right;'>", unsafe_allow_html=True)
    st.markdown("<h1 class='preview-header' style='text-align: center;'>Preview of the document</h1>", unsafe_allow_html=True)
    pdf_container = st.container()

with col1:
    st.markdown("<h1 class='vse-ai'>VŠE AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='study-buddy'>STUDY BUDDY</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left;'>Chat with the AI</h1>", unsafe_allow_html=True)
    
    # Container for the chat messages
    chat_container = st.container()
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Spacer to push the input to the bottom
    st.write('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)
    
    # Chat input at the bottom of col1
    if prompt := st.chat_input("Ask your study buddy"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            # Generate and display AI response
            response, name_file, chat_history = generate_response(prompt, st.session_state.messages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if name_file:
    with open(name_file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
        with col2:
            # Display PDF in container
            with pdf_container:
                pdf_viewer(PDFbyte)
