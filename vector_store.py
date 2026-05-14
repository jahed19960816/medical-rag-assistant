import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Load dataset
df = pd.read_csv("data/medical_abstracts.csv")

# Remove missing rows
df = df.dropna(subset=["Text"])

# Convert to documents
documents = []

for index, row in df.iterrows():

    doc = Document(
        page_content=row["Text"],
        metadata={
            "title": row["Title"],
            "year": row["Year"],
            "doi": row["DOI"]
        }
    )

    documents.append(doc)

# Chunking
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

print(f"Total Chunks: {len(chunks)}")

# Embedding model
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Create FAISS vector database
vectorstore = FAISS.from_documents(
    chunks,
    embeddings
)

# Save vector DB locally
vectorstore.save_local("faiss_index")

print("\nFAISS vector database created successfully!")