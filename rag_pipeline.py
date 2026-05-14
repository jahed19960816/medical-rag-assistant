from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

# Load embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load FAISS vector DB
vectorstore = FAISS.load_local(
    "faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)

# Create retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# Load LLM
llm = ChatGroq(
    groq_api_key="GROQ_API_KEY",
    model_name="llama-3.1-8b-instant"
)

# Create Retrieval QA chain
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever
)

# User question loop
while True:

    query = input("\nAsk Medical Question: ")

    if query.lower() == "exit":
        break

    response = qa_chain.run(query)

    print("\nAnswer:\n")
    print(response)