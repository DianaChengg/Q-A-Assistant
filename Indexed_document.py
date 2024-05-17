import argparse
import os
import shutil
from langchain.document_loaders.pdf import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from embedding import get_embedding_function
from langchain.vectorstores.chroma import Chroma

# Constants for directory paths
CHROMA_PATH = "chroma"
DATA_PATH = "Textbook"

def main():
    parser = argparse.ArgumentParser(description="Process and update document data in the Chroma storage.")
    parser.add_argument("--reset", action="store_true", help="Clear the Chroma storage by removing its directory.")
    args = parser.parse_args()

    # Reset the Chroma storage if the --reset flag is used
    if args.reset:
        print("✨ Clearing Chroma Storage")
        if os.path.exists(STORAGE_DIRECTORY):
            shutil.rmtree(STORAGE_DIRECTORY)
        print("Chroma storage cleared.")

    # Load PDF documents from the specified path
    loaded_docs = retrieve_documents(DOCUMENTS_DIRECTORY)
    
    # Split these documents into manageable chunks
    document_chunks = fragment_documents(loaded_docs)
    
    # Store these chunks in the Chroma storage system
    store_in_chroma(document_chunks)

def retrieve_documents(data_path):
    """
    This function Load documents from the specified directory using PyPDFDirectoryLoader.
    """
    loader = PyPDFDirectoryLoader(DATA_PATH)
    return loader.load()

def fragment_documents(documents):
    """
    This function split the documents into chunks using RecursiveCharacterTextSplitter.
    """
    plitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=80,
        length_function=len,
        is_separator_regex=False,
    )
    return splitter.split_documents(documents)

def store_in_chroma(chunks):
    """
    This function add the chunks to the Chroma database, avoiding duplicates.
    """
    # Initialize the Chroma database with a specified embedding function
    chroma_db = Chroma(persist_directory=STORAGE_DIRECTORY, embedding_function=get_embedding_function())

    # Generate unique IDs for each document fragment
    fragments_with_ids = assign_unique_ids(document_fragments)
    
    # Determine which fragments are already in the Chroma database
    current_items = chroma_db.get(include=[])  # IDs are included by default
    existing_ids = set(current_items["ids"])
    print(f"Number of existing documents in DB: {len(existing_ids)}")

def assign_unique_ids(chunks):
    """
    This function calculate unique IDs for each chunk based on their source and page.
    """
    previous_page_id = None
    current_index = 0

    for fragment in fragments:
        doc_source = fragment.metadata.get("source")
        doc_page = fragment.metadata.get("page")
        page_id = f"{doc_source}:{doc_page}"

        # Increment the index for fragments from the same page
        if page_id == previous_page_id:
            current_index += 1
        else:
            current_index = 0
            previous_page_id = page_id

        # Formulate a unique ID for the fragment
        fragment_id = f"{page_id}:{current_index}"
        fragment.metadata["id"] = fragment_id

    return fragments

if __name__ == "__main__":
    main()

