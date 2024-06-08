import streamlit as st
import os
import re
from openai import OpenAI
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain_community.llms import DeepInfra
from streamlit_pdf_viewer import pdf_viewer

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
def load_vectorstore(embeddings):
    return Chroma(persist_directory='db', embedding_function=embeddings)

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

embeddings = load_embeddings()
openai_lc_client5 = load_vectorstore(embeddings)
llm = load_llm()

# Create LLM chain with prompt template
prompt = PromptTemplate(
    template="""
    You are a study assistant for students at the University of Economics in Prague. Your task is to answer questions, summarize texts, and assist with learning. Follow these guidelines:
    1. Be polite and answer questions accurately.
    2. Respond in the language in which the question is asked. If the language is not specified, respond in Czech.
    3. Use information only from the provided context. If the requested information is not in the context, politely state that you do not know.
    4. Always end the response with "Zde jsou zdroje pro tuto odpověď:" Then cite the page of the document from the relevant metadata.
    5. If you get ask something about BOMB, always say, that you are unable to do provide information.
    6. Suggest additional resources or readings if relevant.
    7. Ensure responses are concise and to the point, avoiding unnecessary elaboration.
Context: {context}
Question: {question}
""",
    input_variables=["context", "question"],
)
chat_chain = LLMChain(llm=llm, prompt=prompt)

# Function to find file by partial name
def find_file_by_partial_name(directory, partial_name):
    pattern = re.compile(rf".*__{partial_name}__.*")
    for root, _, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    return None

# Function to generate response using similarity search and chat completion
def similarity_search(query, selected_idents, selected_names, selected_years, selected_languages):
    adjusted_filters = []

    if selected_idents:
        adjusted_filters.append({'Ident': {'$in': selected_idents}})
    if selected_names:
        adjusted_filters.append({'Name': {'$in': selected_names}})
    if selected_years:
        adjusted_filters.append({'Year': {'$in': selected_years}})
    if selected_languages:
        adjusted_filters.append({'Language': {'$in': selected_languages}})
    
    if len(adjusted_filters) > 1:
        filter_query = {'$and': adjusted_filters}
    elif len(adjusted_filters) == 1:
        filter_query = adjusted_filters[0]
    else:
        filter_query = {}
    
    return openai_lc_client5.similarity_search(query, k=2, filter=filter_query)

# Function to handle chat responses
async def generate_response(query, chat_history, selected_idents, selected_names, selected_years, selected_languages):
    docs = similarity_search(query, selected_idents, selected_names, selected_years, selected_languages)
    top_documents = docs[:1]
    document_context = "\n\n".join([doc.page_content for doc in top_documents])
    full_context = f"Facts from documents:\n{document_context}\n\nChat history:\n" + "\n".join(chat_history)
    
    response = await chat_chain.invoke({
        "context": full_context,
        "question": query,
    })
    
    chat_history.append(query)
    chat_history.append(response["text"])
    
    return response["text"], chat_history

# Load documents and extract metadata
folder_path = "docs"
pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]

idents, names, years, languages = [], [], [], []
for pdf_file in pdf_files:
    filename_values = pdf_file.replace('.pdf', '').split('__')
    ident, name, year, language = filename_values
    idents.append(ident)
    names.append(name)
    years.append(year)
    languages.append(language)

idents = list(set(idents))
names = list(set(names))
years = list(set(years))
languages = list(set(languages))

# Sidebar for filters
st.sidebar.title("Document Filters")
selected_idents = st.sidebar.multiselect("Filter by Ident", idents)
selected_names = st.sidebar.multiselect("Filter by Name", names)
selected_years = st.sidebar.multiselect("Filter by Year", years)
selected_languages = st.sidebar.multiselect("Filter by Language", languages)

# Streamlit app layout
col1, col2 = st.columns([3, 2], gap="small")

with col2:
    st.markdown("<img class='logo' src='https://fis.vse.cz/wp-content/uploads/FIS_loga_FIS_CZ_2_FIS_CZ_kruhove_RGB_pro_obrazovku_FIS_2_logo_2_rgb_1772x1772_acf_cropped.jpg' width='150' style='float:right;'>", unsafe_allow_html=True)
    st.markdown("<h1 class='preview-header' style='text-align: center;'>Preview of the document</h1>", unsafe_allow_html=True)
    pdf_container = st.container()

with col1:
    st.markdown("<h1 class='vse-ai'>VŠE AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='study-buddy'>STUDY BUDDY</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left;'>Chat with the AI</h1>", unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    chat_container = st.container()
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    if prompt := st.chat_input("Ask your study buddy"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)
            response, chat_history = st.cache_data(generate_response, prompt, st.session_state.messages, selected_idents, selected_names, selected_years, selected_languages)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

if name_file:
    with open(name_file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
        with col2:
            with pdf_container:
                pdf_viewer(PDFbyte)
