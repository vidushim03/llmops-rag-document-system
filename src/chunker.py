from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunk_data(docs):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    print("Chunks Created:", len(chunks))
    return chunks