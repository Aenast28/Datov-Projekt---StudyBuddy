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
    adjusted_filters = []

    # Add filters based on selections
    if selected_idents:
        adjusted_filters.append({'Ident': {'$in': selected_idents}})
    
    if selected_names:
        adjusted_filters.append({'Name': {'$in': selected_names}})
    
    if selected_years:
        adjusted_filters.append({'Year': {'$in': selected_years}})
    
    if selected_languages:
        adjusted_filters.append({'Language': {'$in': selected_languages}})
    
    # Combine filters using '$and' to apply all conditions if there are multiple filters
    if len(adjusted_filters) > 1:
        filter_query = {'$and': adjusted_filters}
    elif len(adjusted_filters) == 1:
        filter_query = adjusted_filters[0]  # Use the single filter directly
    else:
        filter_query = {}  # No filters, match all documents
    
    # Perform the similarity search with the adjusted filters
    # Assuming openai_lc_client5 is defined and configured correctly
    return openai_lc_client5.similarity_search(query, filter=filter_query)



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
    
    return response["text"], name_file, chat_history



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
    st.sidebar.write(f"{name}")



###load model
llm = DeepInfra(model_id="mistralai/Mixtral-8x22B-Instruct-v0.1",deepinfra_api_token="hIvZQRN11e1BLIeYghOFCahQYX18uXeY")
llm.model_kwargs = {
    "temperature": 0.4,
    "repetition_penalty": 1.2,
    "max_new_tokens": 500,
    "top_p": 0.9,
}

prompt = PromptTemplate(
    template="""
    You are a study assistant for students at the University of Economics in Prague. Your task is to answer questions, summarize texts, and assist with learning. Follow these guidelines:
    1. Be polite and answer questions accurately.
    2. Respond in the language in which the question is asked. If the language is not specified, respond in Czech.
    3. Use information only from the provided context. If the requested information is not in the context, politely state that you do not know.
    4. At the end, include information about the source of the information, always cite the name of the document and the page.
    5. Always adhere to the maximum token length limit.
    6. Provide examples or explanations to clarify complex concepts.
    7. Offer step-by-step solutions to problems when applicable.
    8. Suggest additional resources or readings if relevant and available in the context.
    9. Use bullet points or numbered lists for clarity when appropriate.
    10. Ensure responses are concise and to the point, avoiding unnecessary elaboration.
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
        margin-top: -2.5em; /* Negative margin to move it higher */
        margin-bottom: 0.2em; /* Reduced margin */
    }
    .study-buddy {
        color: black;
        font-size: 2em; /* Adjusted font size */
        margin-top: -2em; /* Negative margin to move it higher */
        margin-bottom: 0.2em; /* Reduced margin */
    }
    .right-align {
        display: flex;
        justify-content: flex-end;
    }
    .preview-header {
        margin-top: -1.6em; /* Negative margin to move it higher */
    }
    .logo {
        margin-top: -3em; /* Negative margin to move it higher */
    }
    .chat-container {
        height: 650px;
        overflow-y: auto;
        border: 1px solid #ccc; /* Optional: Add a border for better visualization */
        padding: 10px; /* Optional: Add padding for better visualization */
    }
    .resizable {
        display: flex;
        overflow: hidden;
    }
    .resizable > div {
        resize: horizontal;
        overflow: auto;
        border: 1px solid #ccc; /* Optional: for better visualization */
        padding: 10px;
        flex: 1 1 auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# JavaScript for column resizing
st.markdown(
    """
    <script>
    window.addEventListener('DOMContentLoaded', (event) => {
        const col1 = document.querySelectorAll('section.main > div.block-container > div:nth-child(2) > div')[0];
        const col2 = document.querySelectorAll('section.main > div.block-container > div:nth-child(2) > div')[1];
        col1.style.flex = '1 1 auto';
        col2.style.flex = '1 1 auto';
        col1.style.resize = 'horizontal';
        col2.style.resize = 'horizontal';
    });
    </script>
    """,
    unsafe_allow_html=True
)
col1, col2 = st.columns([3, 2], gap="small")

with col2:
    st.markdown("<img class='logo' src='https://fis.vse.cz/wp-content/uploads/FIS_loga_FIS_CZ_2_FIS_CZ_kruhove_RGB_pro_obrazovku_FIS_2_logo_2_rgb_1772x1772_acf_cropped.jpg' width='150' style='float:right;'>", unsafe_allow_html=True)
    st.markdown("<h1 class='preview-header' style='text-align: center;'>Preview of the document</h1>", unsafe_allow_html=True)
    pdf_container = st.container(height=650, border=True)

with col1:
    st.markdown("<h1 class='vse-ai'>VŠE AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='study-buddy'>STUDY BUDDY</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left;'>Chat with the AI</h1>", unsafe_allow_html=True)
    
    # Container for the chat messages
    chat_container = st.container(height=650, border=True)
    with chat_container:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        st.write('<div style="flex-grow: 1;"></div>', unsafe_allow_html=True)

    # Chat input outside of chat_container
    if prompt := st.chat_input("Jak mohu pomoci?"):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Generate and display AI response
        with st.chat_message("assistant"):
            response, name_file, chat_history = generate_response(prompt)
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

if name_file:
        with open(name_file, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
            with col2:
                # Zobrazení PDF v kontejneru
                with pdf_container:
                    pdf_viewer(PDFbyte)
