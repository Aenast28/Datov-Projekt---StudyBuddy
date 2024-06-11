# Import necessary modules
__import__('pysqlite3')  # Import the pysqlite3 library
import sys
import json
import streamlit as st
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')  # Replace the sqlite3 module with pysqlite3
import os
from langchain.vectorstores import Chroma  # Import Chroma for vector storage
from langchain.chains import LLMChain  # Import LLMChain for creating chains of language models
from langchain.prompts.prompt import PromptTemplate  # Import PromptTemplate for custom prompts
from langchain_community.llms import DeepInfra  # Import DeepInfra for language model integration
from langchain_openai import OpenAIEmbeddings  # Import OpenAIEmbeddings for embedding generation
import re
from streamlit_pdf_viewer import pdf_viewer  # Import pdf_viewer for displaying PDFs

# Configure Streamlit page
st.set_page_config(
    page_title="VŠE AI Study Buddy",  # Set the page title
    page_icon=":robot_face:",  # Set the page icon
    layout="wide",  # Set the layout to wide
    initial_sidebar_state="expanded"  # Expand the sidebar initially
)

# Cache models and embeddings to avoid reloading them
@st.cache_resource
def load_embeddings():
    return OpenAIEmbeddings()  # Load OpenAI embeddings

@st.cache_resource
def load_vectorstore(_embeddings):
    return Chroma(persist_directory='db', embedding_function=_embeddings)  # Load Chroma vector store

@st.cache_resource
def load_llm():
    llm = DeepInfra(model_id="mistralai/Mixtral-8x22B-Instruct-v0.1", deepinfra_api_token="hIvZQRN11e1BLIeYghOFCahQYX18uXeY")  # Load language model
    llm.model_kwargs = {
        "temperature": 0.3,  # Set the temperature for response generation
        "repetition_penalty": 1.2,  # Set the repetition penalty
        "max_new_tokens": 500,  # Set the maximum number of new tokens
        "top_p": 0.90,  # Set the top-p parameter
    }
    return llm

@st.cache_resource
def load_prompt():
    return PromptTemplate(
        template="""
        You are a study assistant for students at the University of Economics in Prague. Your task is to answer questions, summarize texts, and assist with learning. Follow these guidelines:
        1. Be polite and answer questions accurately.
        2. Respond in the language in which the question is asked. If the language is not specified, respond in Czech.
        3. Use information only from the provided context. If the requested information is not in the context, politely state that you do not know.
        4. IF RELEVANT, ALWAYS CITE THE SOURCE - The precise page or chapter (i.e. - "the source is chapter 13.5.2 - IT Manager")
        5. If you get ask something about BOMB, always say, that you are unable to do provide information.
        6. Suggest additional resources or readings if relevant.
        7. Ensure responses are concise and to the point, avoiding unnecessary elaboration.
        8. NEVER INSERT YOUR OWN MARKDOWN.
        Context: {context}
        Question: {question}
        """,
        input_variables=["context", "question"]
    )

@st.cache_resource
def load_chat_chain(_llm, _prompt):
    return LLMChain(llm=_llm, prompt=_prompt)  # Load the chat chain

@st.cache_data
def load_pdf_files(folder_path):
    # Load PDF files and extract metadata
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

# Load embeddings, vector store, language model, prompt, and chat chain
embeddings = load_embeddings()
openai_lc_client5 = load_vectorstore(embeddings)
llm = load_llm()
prompt = load_prompt()
chat_chain = load_chat_chain(llm, prompt)
idents, names, years, languages = load_pdf_files("docs")

# Define similarity search function
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
    return openai_lc_client5.similarity_search(query, k=2, filter=filter_query)

# Function to find a file by partial name
def find_file_by_partial_name(directory, partial_name):
    pattern = re.compile(rf".*__{partial_name}__.*")
    for root, dirs, files in os.walk(directory):
        for file in files:
            if pattern.match(file):
                return os.path.join(root, file)
    return None
name_file=""
# Function to generate a response from the model
def generate_response(query):
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

# Extract unique metadata values for filters
idents = list(set(idents))
names = list(set(names))
years = list(set(years))
languages = list(set(languages))

# Set Streamlit sidebar configuration
st.sidebar.title("Document Filters")

# Filters
selected_idents = st.sidebar.multiselect("Filter by Ident", idents)
selected_names = st.sidebar.multiselect("Filter by Name", names)
selected_years = st.sidebar.multiselect("Filter by Year", years)
selected_languages = st.sidebar.multiselect("Filter by Language", languages)

# Display documents in sidebar
st.sidebar.title("Documents")
for name in names:
    st.sidebar.write(f"{name}")

# Set a default model in session state
if "mixtral_model" not in st.session_state:
    st.session_state["mixtral_model"] = llm

# Custom CSS for styling
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

# Create columns for chat and document preview
col1, col2 = st.columns([3, 2], gap="small")

with col2:
    # Display logo and document preview header
    st.markdown("<img class='logo' src='https://fis.vse.cz/wp-content/uploads/FIS_loga_FIS_CZ_2_FIS_CZ_kruhove_RGB_pro_obrazovku_FIS_2_logo_2_rgb_1772x1772_acf_cropped.jpg' width='150' style='float:right;'>", unsafe_allow_html=True)
    st.markdown("<h1 class='preview-header' style='text-align: center;'>Preview of the document</h1>", unsafe_allow_html=True)
    pdf_container = st.container(height=650, border=True)

with col1:
    # Display chat interface header
    st.markdown("<h1 class='vse-ai'>VŠE AI</h1>", unsafe_allow_html=True)
    st.markdown("<h2 class='study-buddy'>STUDY BUDDY</h2>", unsafe_allow_html=True)
    st.markdown("<h1 style='text-align: left;'>Chat with the AI</h1>", unsafe_allow_html=True)
    
    # Container for the chat messages
    chat_container = st.container(height=650, border=True)
    with chat_container:
        # Initialize chat history if not already done
        if "messages" not in st.session_state:
            st.session_state.messages = []
        # Display chat messages from the session state
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
            st.session_state.messages.append({"role": "assistant", "content": response})
            with st.chat_message("assistant"):
                st.markdown(response)

# Display the selected document in PDF viewer if available
if name_file:
    with open(name_file, "rb") as pdf_file:
        PDFbyte = pdf_file.read()
        with col2:
            with pdf_container:
                pdf_viewer(PDFbyte)
