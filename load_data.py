import os
from langchain_community.document_loaders import PyPDFLoader,WebBaseLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

from split_cv import split_cv_sections


cv_p = "/home/malak/Downloads/cellula_week3/Malak_Sobhy_CV_MachineLearning-1.pdf"

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# loading
def load_data(path):
    loader = PyPDFLoader(path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source_file"] = os.path.basename(path)
        doc.metadata["category"] = "CV"  
        
    return docs

# chunking
def chunk_data(doc):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size =300 ,
        chunk_overlap =50,
        separators=["\n\n", "\n", " ", ""]
    )
    chunks =text_splitter.split_documents(doc)
    return chunks

# embedding
def embed_chunks(chunks):
    texts =[chunk.page_content for chunk in chunks]
    embeddings =embedding_model.embed_documents(texts)

    return embeddings

# vector database
def vector_database(chunks):
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        collection_name="malak_data",
        persist_directory="./chroma_db"
    )
    return vector_store



doc =load_data(cv_p)
chunks =split_cv_sections(doc)
vector_store =vector_database(chunks)



# from collections import Counter

# data = vector_store.get(include=["documents", "metadatas"])

# print("Total documents in Chroma:", len(data["ids"]))
# print("Unique document texts:", len(set(data["documents"])))

# counts = Counter(data["documents"])
# duplicates = [text for text, count in counts.items() if count > 1]

# print("Number of duplicated chunks:", len(duplicates))

# for chunk in duplicates[:5]:
#     print("\nDUPLICATE:")
#     print(chunk[:200])


# print("Total IDs:", len(data["ids"]))
# print("Unique IDs:", len(set(data["ids"])))

# for chunk, count in counts.most_common():
#     print(f"Copies: {count} | Text: {chunk[:80]!r}")

# embeddings =embed_chunks(chunks)


# print(f"Number of chunks: {len(chunks)}")

for i, chunk in enumerate(chunks, start=1):
    print(f"\n{'=' * 20} CHUNK {i} {'=' * 20}")
    print(f"Characters: {len(chunk.page_content)}")
    print(chunk.page_content)


# print(f"cv chunked in {len(chunks)}")
# print("first chunk:",chunks[0].page_content)
# print("metadata:" ,chunks[0].metadata)

# print("num of embeddings:", len(embeddings))
# print("embedding dimensions:", len(embeddings[0]))
# print("first embedding:", embeddings[0][:10])


# queries = [
#     "What defenses did Malak use against data poisoning?",
#     "Describe Malak's federated learning project",
#     "How did Malak optimize encrypted inference?",
# ]

# for query in queries:
#     print(f"\n{'=' * 60}")
#     print("QUERY:", query)

#     results = vector_store.similarity_search_with_score(
#         query,
#         k=3
#     )

#     for i, (document, score) in enumerate(results, start=1):
#         print(f"\nResult {i} | Distance: {score:.4f}")
#         print("Section:", document.metadata.get("section"))
#         print("Item:", document.metadata.get("item"))
#         print(document.page_content)