import argparse
import os
import shutil
import speech_recognition as sr
import pyttsx3
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate  # Corrected import
from langchain_community.llms import Ollama

from embedding import get_embedding_function

# Directory path constants
DATABASE_DIRECTORY = "chroma_storage"

# Template for how the AI should structure its answer
ANSWER_TEMPLATE = """
Answer the question based on the above context: {question}
"""

def main():
    # Initialize the speech recognizer and text-to-speech engine
    recognizer = sr.Recognizer()
    tts_engine = pyttsx3.init()

    # Get user question via voice
    user_question = get_voice_input(recognizer)
    if not user_question:
        tts_engine.say("No question detected, please try again.")
        tts_engine.runAndWait()
        return

    # Process the question
    response = process_question(user_question)

    # Output the response using text-to-speech
    tts_engine.say("Here is the response for your question:")
    tts_engine.say(response)
    tts_engine.runAndWait()

def get_voice_input(recognizer):
    """Capture voice input from the user and convert it to text."""
    with sr.Microphone() as source:
        print("Listening for your question...")
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google's speech recognition
        user_question = recognizer.recognize_google(audio)
        print(f"Interpreted question: {user_question}")
        return user_question
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return None

def process_question(user_question: str):
    """Process the user question to find and return an appropriate answer."""
    print("Got it! Your question is: ", user_question)
    
    embedding_function = get_embedding_function()
    chroma_db = Chroma(persist_directory=DATABASE_DIRECTORY, embedding_function=embedding_function)

    search_results = chroma_db.similarity_search_with_score(user_question, k=5)

    compiled_context = "\n\n---\n\n".join([document.page_content for document, _score in search_results])

    prompt_template = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    formatted_prompt = prompt_template.format(context=compiled_context, question=user_question)

    ai_model = Ollama(model="mistral")
    generated_response = ai_model.invoke(formatted_prompt)

    source_documents = [document.metadata.get("id", None) for document, _score in search_results]

    detailed_response = f"Response: {generated_response}\nSources: {source_documents}"
    print(detailed_response)

    return generated_response

if __name__ == "__main__":
    main()
import argparse
import os
import shutil
import speech_recognition as sr
import pyttsx3
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate  # Corrected import
from langchain_community.llms import Ollama

from embedding import get_embedding_function

# Directory path constants
DATABASE_DIRECTORY = "chroma_storage"

# Template for how the AI should structure its answer
ANSWER_TEMPLATE = """
Answer the question based on the above context: {question}
"""

def main():
    # Initialize the speech recognizer and text-to-speech engine
    recognizer = sr.Recognizer()
    tts_engine = pyttsx3.init()

    # Get user question via voice
    user_question = get_voice_input(recognizer)
    if not user_question:
        tts_engine.say("No question detected, please try again.")
        tts_engine.runAndWait()
        return

    # Process the question
    response = process_question(user_question)

    # Output the response using text-to-speech
    tts_engine.say("Here is the response for your question:")
    tts_engine.say(response)
    tts_engine.runAndWait()

def get_voice_input(recognizer):
    """Capture voice input from the user and convert it to text."""
    with sr.Microphone() as source:
        print("Listening for your question...")
        audio = recognizer.listen(source)

    try:
        # Recognize speech using Google's speech recognition
        user_question = recognizer.recognize_google(audio)
        print(f"Interpreted question: {user_question}")
        return user_question
    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
    return None

def process_question(user_question: str):
    """Process the user question to find and return an appropriate answer."""
    print("Got it! Your question is: ", user_question)
    
    embedding_function = get_embedding_function()
    chroma_db = Chroma(persist_directory=DATABASE_DIRECTORY, embedding_function=embedding_function)

    search_results = chroma_db.similarity_search_with_score(user_question, k=5)

    compiled_context = "\n\n---\n\n".join([document.page_content for document, _score in search_results])

    prompt_template = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    formatted_prompt = prompt_template.format(context=compiled_context, question=user_question)

    ai_model = Ollama(model="mistral")
    generated_response = ai_model.invoke(formatted_prompt)

    source_documents = [document.metadata.get("id", None) for document, _score in search_results]

    detailed_response = f"Response: {generated_response}\nSources: {source_documents}"
    print(detailed_response)

    return generated_response

if __name__ == "__main__":
    main()
