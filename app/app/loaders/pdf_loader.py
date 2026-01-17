from langchain_community.document_loaders import PyPDFLoader
from typing import List
from langchain.schema import Document


def load_pdf(path: str) -> List[Document]:
    loader = PyPDFLoader(path)
    docs = loader.load()

    for doc in docs:
        doc.metadata["source"] = path
        doc.metadata["page"] = doc.metadata.get("page", 0)

    return docs
