import argparse
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from embedding import get_embedding_function

# Constants for the storage path and answer format
DATABASE_DIRECTORY = "chroma_storage"
ANSWER_TEMPLATE = """
Answer the question based on the above context: {question}
"""

def main():
    # Interactive user input for the question
    user_question = input("Hello! What's your question? ").strip()

    # Check if the user provided an empty question and provide a fallback response
    if not user_question:
        print("You didn't enter a question. Please try again.")
        return

    process_question(user_question)

def process_question(user_question: str):
    # Display the received question
    print("Got it! Your question is: ", user_question)
    
    # Initialize embedding function and Chroma database
    embedding_function = get_embedding_function()
    chroma_db = Chroma(persist_directory=DATABASE_DIRECTORY, embedding_function=embedding_function)

    # Search the database for similar contexts
    search_results = chroma_db.similarity_search_with_score(user_question, k=5)

    # Check if there are no search results and provide a fallback
    if not search_results:
        print("No relevant context was found for your question. Please try asking something else.")
        return

    # Compile the context information from search results
    compiled_context = "\n\n---\n\n".join([document.page_content for document, _score in search_results])
    
    if not compiled_context.strip():
        print("The context for your question is empty. Please try a different question.")
        return

    # Prepare the prompt using the template
    prompt_template = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    formatted_prompt = prompt_template.format(context=compiled_context, question=user_question)
    print(formatted_prompt)

    # Use the Ollama model to generate an answer
    ai_model = Ollama(model="mistral")
    try:
        generated_response = ai_model.invoke(formatted_prompt)
    except Exception as e:
        print(f"Failed to generate a response due to an error: {e}")
        return

    # List of source documents
    source_documents = [document.metadata.get("id", None) for document, _score in search_results]
    
    # Format and display the final response
    detailed_response = f"Response: {generated_response}\nSources: {source_documents}"
    print("Here is the response for your question:")
    print(detailed_response)

if __name__ == "__main__":
    main()

