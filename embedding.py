from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain_community.embeddings.bedrock import BedrockEmbeddings
import langchain_community.embeddings.ollama as ollama


def get_embedding_function():
    """
    Initializes and returns an OllamaEmbeddings model.

    This function specifically sets up the 'OllamaEmbeddings' model using
    the 'mxbai-embed-large' model configuration. This model is often used
    for generating embeddings from text, which can then be used for various
    NLP tasks such as document similarity, clustering, and more.

    Returns:
        OllamaEmbeddings: A model instance initialized with the 'mxbai-embed-large' model.
    """
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")
    return embeddings
