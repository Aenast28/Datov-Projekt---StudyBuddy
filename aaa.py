from typing import List, Dict
from openai import OpenAI
import chromadb  # Assuming chromadb is a hypothetical module for this example
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
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
files = os.listdir("C:\\Users\\scott\\Downloads\\docs")

for file_path in files:
    elements = partition(file_path, include_page_breaks=False)
    filtered_elements = [element for element in elements if not isinstance(element, (Header, Footer))]
    #Clean line break dashes, bullets, whitespaces, non ascii characters and punctuation from elements
    
    #Custom function to remove line break dashes
    remove_line_break_dashes = lambda text: re.sub(r"- ", "", text)
    #Custom function to remove multiple periods (more than 4)
    remove_multiple_periods = lambda text: re.sub(r"\.{4,}", ".", text)
    
    # Define a function to clean the text content of an element
    def clean_element_text(element):
        if hasattr(element, 'text'):
            # Extract text
            text = element.text
    
            # Apply cleaning functions to the text
            text = remove_line_break_dashes(text)
            text = remove_multiple_periods(text)
            text = clean(text, bullets=True, extra_whitespace=True, dashes=True)
    
            # Re-assign cleaned text back to the element
            element.text = text
    
    # Apply cleaning to each element
    for element in filtered_elements:
        clean_element_text(element)
    chunks = chunk_by_title(filtered_elements, max_characters=1500, new_after_n_chars=1000, overlap=100, overlap_all=False, multipage_sections=True, combine_text_under_n_chars=350)
    # Define a Document class with a page_content attribute
    class Document:
        def __init__(self, page_content, metadata):
            self.page_content = page_content
            self.metadata = metadata
    
    def transform_chunks(chunks):
        corrected_chunks = []
        for chunk in chunks:
            # Extrahovat textový obsah z každého chunku
            if hasattr(chunk, 'text'):
                content = chunk.text
            else:
                content = str(chunk)  # Zajistit, že obsah je řetězec, pokud není atribut text
    
            # Extrahovat metadata
            # Lambda function to deconstruct the filename
            if hasattr(chunk.metadata, 'filename'):
                # Extract values from filename
                filename_values = chunk.metadata.filename.split('__')
                ident, name, year, language = filename_values[0], filename_values[1], filename_values[2], filename_values[3]
    
                # Build metadata dictionary
                metadata = {
                    "Ident": ident,
                    "Name": name,
                    "Year": year,
                    "Language": language,
                    "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else None
                }
            else:
                # If filename is not present, only include page_number if available
                metadata = {
                    "Ident": None,
                    "Name": None,
                    "Year": None,
                    "Language": None,
                    "page_number": chunk.metadata.page_number if hasattr(chunk.metadata, 'page_number') else None
                }
    
            corrected_chunks.append(Document(content, metadata))
    
        return corrected_chunks
    
    
    corrected_chunks = transform_chunks(chunks)
    os.environ["OPENAI_API_KEY"] = ""
    
    
    embeddings = OpenAIEmbeddings()
    persist_directory = 'db'
    openai_lc_client5 = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    openai_lc_client5.add_documents(corrected_chunks)
    openai_lc_client5.persist()
