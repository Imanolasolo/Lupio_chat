import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI 
import os
import base64
from htmlTemplates import *

# Set your name for the AIProfileVCard
name = 'LUPIO'

# Function to extract text from a PDF file
def get_pdf_text(pdf_path):
    pdf_reader = PdfReader(pdf_path)
    text = ""
    # Iterate through each page and extract text
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

# Function to split the extracted text into manageable chunks for processing
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to generate a vector store using the text chunks
def get_vector_store(text_chunks):
    embeddings = OpenAIEmbeddings(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to create a conversational chain using the vector store
def get_conversation_chain(vector_store):
    llm = ChatOpenAI(openai_api_key=st.secrets["OPEN_AI_APIKEY"])
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to handle user input and generate responses
def handle_user_input(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    # Display the conversation history
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", msg.content), unsafe_allow_html=True)

# Main function to run the Streamlit app
def main():
    st.set_page_config(page_title=name, page_icon=":computer:", layout="centered")

    # Define CSS for the custom container and input box
    st.markdown("""
        <style>
        .custom-container {
            display: flex;
            justify-content: center;
            align-items: center;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            max-width: 600px;
            margin: 20px auto;
        }
        .custom-container img {
            max-width: 100%;
            height: auto;
        }
        .custom-text-input {
            padding: 10px;
            border: 2px solid #ddd;
            border-radius: 10px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            width: 100%;
        }
        </style>
    """, unsafe_allow_html=True)

    # Display the custom container with a local image
    image_path = 'Logo.png'
    if os.path.exists(image_path):
        st.markdown(f'''
            <div class="custom-container">
                <img src="data:image/png;base64,{get_base64_of_bin_file(image_path)}" alt="Logo">
            </div>
        ''', unsafe_allow_html=True)
    else:
        st.write("Image file not found")

    # Display profile picture and description
    col1, col2 = st.columns(2)
    with col1:
        st.image('developers.gif')
    with col2:
        description = """
            Lupio fue fundado hace más de 12 años con el objetivo de potenciar la presencia digital de las empresas a través del desarrollo de soluciones digitales innovadoras. Nos especializamos en crear herramientas efectivas para el crecimiento empresarial y ofrecemos asesoría y soporte estratégico para garantizar el éxito y la optimización de nuestros clientes en el entorno digital.
        """
        st.markdown(description)

    # Section for interacting with the AI chatbot
    st.title('Chatea con LUPIO IA')
    st.write("### Solicita información sobre nuestros servicios, consultas de tecnología, asesoría y más..")

    # Process the PDF file to be used as context for the chatbot
    pdf_path = os.path.join(os.getcwd(), "Base_conocimiento_Lupio.pdf")
    pdf_text = get_pdf_text(pdf_path)
    text_chunks = get_text_chunks(pdf_text)
    vector_store = get_vector_store(text_chunks)
    conversation_chain = get_conversation_chain(vector_store)

    # Store the conversation chain and history in session state
    st.session_state.conversation = conversation_chain
    st.session_state.chat_history = []

    # Input box for user questions with custom styling
    user_question = st.text_input(
        "Hola, soy Lupio IA, ¿en qué te puedo ayudar?",
        key="user_input",
        placeholder="Escribe tu mensaje aquí..."
    )

    # Add styling to the input box (shadow and border)
    st.markdown("""
        <style>
        div[role="textbox"] > input {
            border: 2px solid #ddd !important;
            border-radius: 5px;
            box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        }
        </style>
    """, unsafe_allow_html=True)

    # Handle user input
    if user_question:
        handle_user_input(user_question)

# Function to encode image as base64 to set as background
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Run the main function if the script is executed
if __name__ == "__main__":
    main()
