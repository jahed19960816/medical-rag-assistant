import streamlit as st
import json
import os
import tempfile
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.chains import ConversationalRetrievalChain

#--------------------------
#CREATE CHAT FILE VARIABLE
#---------------------------

CHAT_HISTORY_FILE = "chat_history.json"

# -----------------------------------
# PAGE CONFIG
# -----------------------------------

st.set_page_config(
    page_title="Medical RAG Assistant",
    page_icon="🩺",
    layout="wide"
)

# -----------------------------------
# SIDEBAR
# -----------------------------------

with st.sidebar:

    st.title("🩺 Medical AI Assistant")

    st.markdown("---")

    st.markdown("""
    ### About
    This AI assistant answers medical research questions using:

    - LangChain
    - FAISS
    - HuggingFace Embeddings
    - Groq LLM
    - Retrieval-Augmented Generation (RAG)
    """)

    st.markdown("---")

    st.markdown("### Built By")
    st.write("Jahed Anwar")
    if st.button("Clear Chat"):

        st.session_state.messages = []

        st.rerun()

# -----------------------------------
# MAIN TITLE
# -----------------------------------

    st.markdown("---")

    uploaded_file = st.file_uploader(
        "Upload Medical PDF",
        type="pdf"
    )

st.title("🩺 Medical Research Assistant")

st.caption("Ask questions from medical research papers using AI + RAG")

# -----------------------------------
# SESSION STATE FOR CHAT HISTORY
# -----------------------------------

if "messages" not in st.session_state:

    if os.path.exists(CHAT_HISTORY_FILE):

        with open(CHAT_HISTORY_FILE, "r") as file:

            st.session_state.messages = json.load(file)

    else:

        st.session_state.messages = []

# -----------------------------------
# LOAD EMBEDDINGS
# -----------------------------------

@st.cache_resource
def load_vectorstore():

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )

    return vectorstore

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vectorstore = load_vectorstore()
# -----------------------------------
# PDF Upload Processing
# -----------------------------------

if uploaded_file:

    with st.spinner("Processing PDF..."):

        # Save uploaded PDF temporarily
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=".pdf"
        ) as tmp_file:

            tmp_file.write(uploaded_file.read())

            temp_pdf = tmp_file.name

        # Load PDF
        loader = PyPDFLoader(temp_pdf)

        docs = loader.load()

        # Split PDF into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        pdf_chunks = text_splitter.split_documents(docs)

        if len(pdf_chunks) == 0:

            st.error("No readable text found in PDF.")

            st.stop()

        # Create temporary vector DB
        vectorstore = FAISS.from_documents(
            pdf_chunks,
            embeddings
        )

        st.success("PDF processed successfully!")

# -----------------------------------
# RETRIEVER
# -----------------------------------

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# -----------------------------------
# LOAD LLM
# -----------------------------------

llm = ChatGroq(
    groq_api_key=st.secrets["GROQ_API_KEY"],
    model_name="llama-3.1-8b-instant"
)
# ------------------------------------
# CREATE MEMORY
# ------------------------------------

memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer"
)
# -----------------------------------
# QA CHAIN
# -----------------------------------

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True
)
# -----------------------------------
# DISPLAY CHAT HISTORY
# -----------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

# -----------------------------------
# CHAT INPUT
# -----------------------------------

prompt = st.chat_input("Ask your medical question...")

# -----------------------------------
# USER QUESTION
# -----------------------------------

if prompt:

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # Save user message to JSON
    with open(CHAT_HISTORY_FILE, "w") as file:

        json.dump(st.session_state.messages, file)

    # AI response
    with st.spinner("Searching medical research papers..."):

        response = qa_chain.invoke(
            {"question": prompt}
        )

        answer = response["answer"]

        source_docs = response["source_documents"]

    # Save AI response
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

    # Save assistant response to JSON
    with open(CHAT_HISTORY_FILE, "w") as file:

        json.dump(st.session_state.messages, file)

    # Show assistant response
    with st.chat_message("assistant"):

        st.markdown(answer)

        # Sources
        with st.expander("📚 Source Documents"):

            for doc in source_docs:

                st.markdown(
                    f"""
                    <div style="
                        padding:15px;
                        border-radius:10px;
                        background-color:#262730;
                        margin-bottom:15px;
                    ">

                    <h4>{doc.metadata.get('title')}</h4>

                    <p><b>Year:</b> {doc.metadata.get('year')}</p>

                    <p><b>DOI:</b> {doc.metadata.get('doi')}</p>

                    </div>
                    """,
                    unsafe_allow_html=True
                )

    st.rerun()
       

