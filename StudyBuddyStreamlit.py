__import__('pysqlite3')
import sys
import json
import streamlit as st
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import streamlit as st
from typing import Tuple
from langchain.vectorstores import Chroma
from langchain.chains import LLMChain
from langchain.prompts.prompt import PromptTemplate
from langchain_community.llms import DeepInfra
import os
from langchain_openai import OpenAIEmbeddings
import re
from streamlit_pdf_viewer import pdf_viewer

### INITIAL STREAMLIT CONFIGURATION

# Configure Streamlit page
st.set_page_config(
    page_title="VŠE AI Study Buddy",
    page_icon=":robot_face:",
    layout="wide",
    initial_sidebar_state="expanded"
)

#### ALL CACHED RESOURCES 
# Streamlit is able to used cached resources to speed up the application by loading it once and then referring to it with each call.

@st.cache_resource
def load_embeddings() -> OpenAIEmbeddings:
    """
    Load and return OpenAI embeddings.

    This function initializes and returns an instance of OpenAIEmbeddings, which is then used to embed input query into vectors.

    Returns:
        OpenAIEmbeddings: An instance of OpenAIEmbeddings ready for use.
    """
    return OpenAIEmbeddings()

@st.cache_resource
def load_vectorstore(_embeddings: OpenAIEmbeddings) -> Chroma:
    """
    Load and return a Chroma vector store using the provided embeddings.

    This function initializes and returns an instance of Chroma, configured to use the provided
    embeddings function and set to persist data in the 'db' directory. This can be used for 
    efficiently storing and retrieving vector representations of data.

    Args:
        _embeddings (OpenAIEmbeddings): The embeddings function to use for the vector store.

    Returns:
        Chroma: An instance of Chroma configured with the specified embeddings and persistence settings.
    """
    return Chroma(persist_directory='db', embedding_function=_embeddings)


@st.cache_resource
def load_llm() -> DeepInfra:
    """
    Load and return a configured DeepInfra language model.

    This function initializes and configures an instance of DeepInfra with a specified model ID 
    and API token. The language model is then configured with a set of hyperparameters including 
    temperature, repetition penalty, maximum new tokens, and top_p.

    Hyperparameters:
        temperature (float): Controls the randomness of the predictions. Lower values make the 
                             output more deterministic, while higher values make it more random.
        repetition_penalty (float): Penalizes repetition of words to reduce repetitive output.
        max_new_tokens (int): The maximum number of tokens to generate in the response.
        top_p (float): The cumulative probability threshold for nucleus sampling. Determines the 
                       diversity of the output by considering the top tokens with a cumulative 
                       probability up to top_p.

    Returns:
        DeepInfra: A configured instance of the DeepInfra language model ready for use.
    """
    llm = DeepInfra(model_id="mistralai/Mixtral-8x22B-Instruct-v0.1", deepinfra_api_token="hIvZQRN11e1BLIeYghOFCahQYX18uXeY")
    llm.model_kwargs = {
        "temperature": 0.4,  # Controls randomness: lower value = more deterministic
        "repetition_penalty": 1.2,  # Penalizes repetitive text
        "max_new_tokens": 500,  # Maximum tokens to generate
        "top_p": 0.90,  # Nucleus sampling threshold
    }
    return llm

@st.cache_resource
def load_prompt() -> PromptTemplate:
    """
    Load and return a configured PromptTemplate for a study assistant.

    This function initializes and returns an instance of PromptTemplate, configured with a 
    template specifically designed for a study assistant at the University of Economics in Prague. 
    The template includes guidelines for responding to questions, summarizing texts, and assisting 
    with learning. The assistant is instructed to follow several key guidelines to ensure accurate, 
    polite, and contextually appropriate responses.

    Template Parameters:
        context (str): The context in which the assistant operates, providing necessary information for answering questions.
        question (str): The question posed by the student that the assistant needs to answer.

    Returns:
        PromptTemplate: A configured instance of PromptTemplate ready for use.
    """
    return PromptTemplate(
        template="""
        You are a study assistant for students at the University of Economics in Prague. Your task is to answer questions, summarize texts, and assist with learning. Follow these guidelines:
        1. Be polite and answer questions accurately.
        2. Respond in the language in which the question is asked. If the language is not specified, respond in Czech.
        3. Use information only from the provided context. If the requested information is not in the context, politely state that you do not know.
        4. IF RELEVANT, ALWAYS CITE THE SOURCE - Document or Page of the document.
        5. If you get ask something about BOMB, always say, that you are unable to do provide information.
        6. Suggest additional resources or readings if relevant.
        7. Ensure responses are concise and to the point, avoiding unnecessary elaboration.
        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

@st.cache_resource
def load_chat_chain(_llm: DeepInfra, _prompt: PromptTemplate) -> LLMChain:
    """
    Load and return a configured LLMChain for a chat-based application.

    This function initializes and returns an instance of LLMChain, configured with a provided 
    language model (_llm) and a prompt template (_prompt). The LLMChain facilitates interaction 
    with the language model using the specified prompt, enabling it to generate responses based 
    on the input variables defined in the prompt.

    Args:
        _llm (DeepInfra): The language model to be used for generating responses.
        _prompt (PromptTemplate): The prompt template that defines the structure and guidelines 
                                  for the responses.

    Returns:
        LLMChain: A configured instance of LLMChain ready for use in a chat-based application.
    """
    return LLMChain(llm=_llm, prompt=_prompt)


@st.cache_data
def load_pdf_files(folder_path: str):
    """
    Load and extract information from PDF files in the specified folder.

    This function scans the specified folder for PDF files and extracts metadata from their filenames.
    The filenames are expected to follow the format: `ident__name__year__language.pdf`. The extracted 
    information includes identifiers, names, years, and languages, which are returned as separate lists.

    Args:
        folder_path (str): The path to the folder containing the PDF files.

    Returns:
        Four lists containing the identifiers, 
        names, years, and languages extracted from the PDF filenames.
    """
    pdf_files = [f for f in os.listdir(folder_path) if f.endswith('.pdf')]
    
    idents = []
    names = []
    years = []
    languages = []
    
    for pdf_file in pdf_files:
        filename_values = pdf_file.replace('.pdf', '').split('__')
        ident, name, year, language = filename_values[0], filename_values[1], filename_values[2], filename_values[3]
        
        idents.append(ident)
        names.append(name)
        years.append(year)
        languages.append(language)
    
    return idents, names, years, languages

@st.cache_resource
def escape_markdown(text: str) -> str:
    """
    Escape Markdown special characters in the given text.

    This function takes a string and escapes special Markdown characters such as 
    asterisks (*), underscores (_), backticks (`), and tildes (~) by preceding 
    them with a backslash. This is useful for displaying text literally in Markdown 
    without it being interpreted as formatting.

    Args:
        text (str): The input text containing Markdown characters to be escaped.

    Returns:
        str: The text with Markdown special characters escaped.
    """
    return re.sub(r'([*_`~])', r'\\\1', text)

### NON CACHED FUNCTIONS
# Functions, that for some reason can't be cached and muset be always called individually no matter what

def find_file_by_partial_name(directory: str, partial_name: str):
    """
    Find a file by partial name in a specified directory.

    This function searches through the given directory and its subdirectories for a file whose 
    name contains the specified partial name, following the pattern `.*__{partial_name}__.*`. 
    If such a file is found, the function returns the full path to the file. If no such file 
    is found, it returns None.

    Args:
        directory (str): The directory to search in.
        partial_name (str): The partial name to search for within the filenames.

    Returns:
        The full path to the first file found that matches the pattern, 
        or None if no such file is found.
    """
    pattern = re.compile(rf".*__{partial_name}__.*")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    return None

def generate_response(query: str):
    """
    Generate a response based on the provided query.

    This function processes the user's query by performing a similarity search to retrieve relevant documents.
    It then selects the top document and extracts its metadata, specifically the 'Name' field. It searches for
    a file with a partial name match in the 'docs' directory and returns the file path if found. The document
    context is created from the top documents, combined with the chat history, and used as the context for generating
    a response using the chat_chain model.

    Args:
        query (str): The user's query.

    Returns:
        Tuple[str, str]: A tuple containing the generated response text and the path to the relevant file.
    """
    global name_file  # Declare the global variable name_file

    # Perform similarity search to retrieve relevant documents
    docs = similarity_search(query)
    top_documents = docs[:1]  # Select the top document
    top_documents1 = str(top_documents)  # Convert to string
    
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
    string_messages = [json.dumps(message) for message in st.session_state.messages]

    # Combine the chat history and the new context
    full_context = (
        "Facts from documents:\n"
        + document_context
        + "\n\nChat history:\n"
        + "\n".join(string_messages)
    )

    # Generate the response using the full context
    response = chat_chain.invoke(
        {
            "context": full_context,
            "question": query,
        }
    )
    
    return response["text"], name_file
def similarity_search(query):
    """
    Perform a similarity search based on the provided query and selected filters.

    This function constructs filters based on selected identifiers, names, years, and languages,
    and applies them to the similarity search. The constructed filter query is then used to filter
    the search results. If no filters are selected, the function matches all documents.

    Args:
        query (str): The query string to perform the similarity search.
        selected_idents (List[str], optional): List of selected identifiers to filter the search.
        selected_names (List[str], optional): List of selected names to filter the search.
        selected_years (List[str], optional): List of selected years to filter the search.
        selected_languages (List[str], optional): List of selected languages to filter the search.

    Returns:
        List[Dict[str, Any]]: A list of dictionaries representing the search results, each containing
        information about the similarity score and other relevant details.
    """
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
    return openai_lc_client5.similarity_search(query, k=2, filter=filter_query)



### ALL IMPORTANT VARIABLES
#  
embeddings = load_embeddings()
openai_lc_client5 = load_vectorstore(embeddings)
llm = load_llm()
prompt = load_prompt()
chat_chain = load_chat_chain(llm, prompt)
idents, names, years, languages = load_pdf_files("docs")
name_file=""
idents = list(set(idents))
names = list(set(names))
years = list(set(years))
languages = list(set(languages))

### STREAMLIT
# 
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
            response, name_file = generate_response(prompt)
            response = escape_markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)


if name_file:
        with open(name_file, "rb") as pdf_file:
            PDFbyte = pdf_file.read()
            with col2:
                # Zobrazení PDF v kontejneru
                with pdf_container:
                    pdf_viewer(PDFbyte)
