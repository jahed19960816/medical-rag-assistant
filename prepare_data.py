import pandas as pd
from langchain.schema import Document

# Load dataset
df = pd.read_csv("data/medical_abstracts.csv")

# Remove missing text rows
df = df.dropna(subset=["Text"])

# Convert rows into LangChain documents
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

# Show first document
print(documents[0])

# Total documents
print(f"\nTotal Documents: {len(documents)}")