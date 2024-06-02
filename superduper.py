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
import chromadb  # Assuming chromadb is a hypothetical module for this example
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
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
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean
from unstructured.documents.elements import Header, Footer
import string
from streamlit_pdf_viewer import pdf_viewer

#### LOAD DOC ########################################
###################################################
# Nastavit cestu k složce s PDF soubory
folder_path = "docs"
# Získat seznam všech PDF souborů ve složce
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

# Seznamy pro uložení jednotlivých hodnot
idents = []
names = []
years = []
languages = []
# Načíst obsah všech PDF souborů a extrahovat hodnoty z názvů
for pdf_file in pdf_files:
    # Extrahovat hodnoty z názvu souboru
    filename_values = pdf_file.replace('.pdf', '').split('__')
    ident, name, year, language = filename_values[0], filename_values[1], filename_values[2], filename_values[3]
    
    # Uložit jednotlivé hodnoty do seznamů
    idents.append(ident)
    names.append(name)
    years.append(year)
    languages.append(language)



#############################
#chatbot streamlit a funkce ##################
#############################
# Assume you have a similarity search function defined, which searches documents based on a query
def similarity_search(query):
    # This is a placeholder for your similarity search function.
    # Replace it with the actual implementation.
    return openai_lc_client5.similarity_search(query)

# Function to generate response using similarity search and chat completion
chat_history=[]
import re

# Definice proměnné name_file
name_file = ""

import re
import os

def find_file_by_partial_name(directory, partial_name):
    pattern = re.compile(rf".*__{partial_name}__.*")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    return None

def generate_response(query):
    global name_file  # Deklarace globální proměnné name_file
    # Perform similarity search to retrieve relevant documents
    docs = similarity_search(query)
    top_documents = docs[:3]  # Select the top three documents
    top_documents1 = str(top_documents)  # Převést vstup na řetězec
    
    # Clear the current string in name_file
    name_file = ""
    
    # Search for the "Name" in the metadata of the documents
    match = re.search(r"metadata=\{.*'Name': '([^']*)'.*\}", top_documents1)
    if match:
        partial_name = match.group(1)
        directory = r"docs"
        file_path = find_file_by_partial_name(directory, partial_name)
        if file_path:
            name_file = file_path
    
    # Create the context from the top documents
    document_context = "\n\n".join([doc.page_content for doc in top_documents])
    
    # Combine the chat history and the new context
    full_context = (
        "Facts from documents:\n"
        + document_context
        + "\n\nChat history:\n"
        + "\n".join(chat_history)
    )

    # Generate the response using the full context
    response = chat_chain.invoke(
        {
            "context": full_context,
            "question": query,
        }
    )
    
    # Store the query and response in chat history
    chat_history.append(f"User: {query}")
    chat_history.append(f"Assistant: {response['text']}")
    
    return response["text"], name_file



# Extract unique metadata values for filters
idents = list(set(idents))
names = list(set(names))
years = list(set(years))
languages = list(set(languages))

# Streamlit app layout with theme
st.set_page_config(
    page_title="VŠE AI Study Buddy",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)


#os.environ["OPENAI_API_KEY"] ==st.secrets["OPENAI_API_KEY"]

embeddings = OpenAIEmbeddings()
persist_directory = 'db'
openai_lc_client5 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-color: #00957d;
    }
    .css-1d391kg {
        color: green;
    }
    
    </style>
    """,
    unsafe_allow_html=True
)

st.sidebar.title("Document Filters")

# Filters
selected_idents = st.sidebar.multiselect("Filter by Ident", idents)
selected_names = st.sidebar.multiselect("Filter by Name", names)
selected_years = st.sidebar.multiselect("Filter by Year", years)
selected_languages = st.sidebar.multiselect("Filter by Language", languages)

# Filter documents based on selections

st.sidebar.title("Documents")
# Iterate over the unique names and write them to the sidebar
for name in names:
    st.sidebar.write(f"Title: {name}")



###load model
llm = DeepInfra(model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",deepinfra_api_token="hIvZQRN11e1BLIeYghOFCahQYX18uXeY")
llm.model_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1.2,
    "max_new_tokens": 350,
    "top_p": 0.9,
}

prompt = PromptTemplate(
    template="""Jsi pomocník se studiem na Vysoké škole ekonomické v Praze. Tvým úkolem je studentům odpovídat na otázky, sumarizovat texty a pomáhat s učením. Následuj tyto pokyny.
    1) Buď zvořilý a odpovídej přesně na položené otázky.
    2) Odpovídej v jazyce, ve kterém je položena otázka. Pokud není jazyk specifikovaný tak odpovídej v českém jazyce.
    3) Ber informace pouze z přidaného kontextu a pokud v něm nebudou informace požadované v otázce, zdvořile řekni, že nevíš.
    4) Na konec přidej informaci o zdroji informací, tedy název dokumentu, kapitolu a stránku ze které jsi čerpal.
Context: {context}
Question: {question}
""",
    input_variables=["context", "question"],
)
chat_chain = LLMChain(llm=llm, prompt=prompt)

# Set a default model
if "mixtral_model" not in st.session_state:
    st.session_state["mixtral_model"] = llm


st.markdown(
    """
    <style>
    .vse-ai {
        color: #00957d;
        font-size: 2.5em; /* Adjusted font size */
        margin-bottom: 0.2em; /* Reduced margin */
    }
    .study-buddy {
        color: black;
        font-size: 2em; /* Adjusted font size */
        margin-top: 0;
        margin-bottom: 0.2em; /* Reduced margin */
    }
    .right-align {
        display: flex;
        justify-content: flex-end;
    }
    </style>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns([3, 2])

with col2:
    st.image("logo_fis.jpg", width=135, output_format='auto')
    st.markdown("<style>div.stImage {display: flex; justify-content: flex-end;}</style>", unsafe_allow_html=True)
    pdf_container = st.container(height=650, border=True)
    with pdf_container:
        st.markdown("<div style='border: 2px solid black; padding: 10px;'><h1 style='text-align: center;'>Preview of the document</h1></div>", unsafe_allow_html=True)


with col1:
    st.markdown("<h1 class='vse-ai'>VŠE AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='study-buddy'>STUDY BUDDY</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left;'>Chat with the AI</h1>", unsafe_allow_html=True)
    chat_container = st.container(height=650,border=True)
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        if prompt := st.chat_input("Jak mohu pomoci?"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})
            # Display user message in chat message container
            with st.chat_message("user"):
                st.markdown(prompt)
            with st.chat_message("assistant"):
                response, name_file = generate_response(prompt)
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if name_file:
        with open(name_file, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
            with col2:
                # Zobrazení PDF v kontejneru s rámečkem
                with pdf_container:
                    st.markdown("<div style='border: 2px solid black; padding: 10px;'>", unsafe_allow_html=True)  # Začátek kontejneru s rámečkem
                    pdf_viewer(PDFbyte)
                    st.markdown("</div>", unsafe_allow_html=True)  # Konec kontejneru s rámečkem


