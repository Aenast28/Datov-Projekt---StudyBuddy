__import__('pysqlite3')
import sys
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
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.auto import partition
from unstructured.cleaners.core import clean
from unstructured.documents.elements import Header, Footer
import string

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
def generate_response(query):
    # Perform similarity search to retrieve relevant documents
    docs = similarity_search(query)
    top_documents = docs[:3]  # Select the top three documents

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
    
    return response["text"]
# Extract unique metadata values for filters
idents = list(set(idents))
names = list(set(names))
years = list(set(years))
languages = list(set(languages))

# # Function to filter documents by selected metadata
# def filter_documents(selected_idents: List[str], selected_names: List[str], selected_years: List[str], selected_languages: List[str]) -> List[Document]:
#     return [
#         doc for doc in corrected_chunks
#         if (not selected_idents or doc.metadata["Ident"] in selected_idents) and
#            (not selected_names or doc.metadata["Name"] in selected_names) and
#            (not selected_years or doc.metadata["Year"] in selected_years) and
#            (not selected_languages or doc.metadata["Language"] in selected_languages)
#     ]

# Streamlit app layout with theme
st.set_page_config(
    page_title="VŠE AI Study Buddy",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.write(
    os.environ["OPENAI_API_KEY"] ==st.secrets["OPENAI_API_KEY"],
)


embeddings = OpenAIEmbeddings()
persist_directory = 'db'
openai_lc_client5 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

st.markdown(
    """
    <style>
    .css-18e3th9 {
        background-color: #f0f0f0;
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
#filtered_documents = filter_documents(selected_idents, selected_names, selected_years, selected_languages)

st.sidebar.title("Documents")

# Use a set comprehension to get unique names directly from the documents
unique_names = {doc.metadata['Name'] for doc in filtered_documents}

# Iterate over the unique names and write them to the sidebar
for name in unique_names:
    st.sidebar.write(f"Title: {name}")

st.title("VŠE AI Study Buddy")
st.image("logo_fis.jpg", width=150)

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

# Chat window
st.header("Chat with the AI")

# Set a default model
if "mixtral_model" not in st.session_state:
    st.session_state["mixtral_model"] = llm

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# download document button
with open(file_path, "rb") as pdf_file:
    PDFbyte = pdf_file.read()

st.download_button(label="Download last cited document",
                    data=PDFbyte,
                    file_name="name.pdf",
                    mime='application/octet-stream')

# Accept user input
if prompt := st.chat_input("Jak mohu pomoci?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)
    with st.chat_message("assistant"):
        response = generate_response(prompt)
        st.markdown(response)
    st.session_state.messages.append({"role": "assistant", "content": response})
