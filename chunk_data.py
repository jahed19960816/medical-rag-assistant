import pandas as pd
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load dataset
df = pd.read_csv("data/medical_abstracts.csv")

# Remove missing rows
df = df.dropna(subset=["Text"])

# Convert to LangChain documents
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

# Output
print(f"Total Documents: {len(documents)}")
print(f"Total Chunks: {len(chunks)}")

print("\nFirst Chunk:\n")
print(chunks[0].page_content[:1000])