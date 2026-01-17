import os
import warnings
import logging

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TRANSFORMERS_VERBOSITY"] = "error"

warnings.filterwarnings("ignore")
logging.getLogger().setLevel(logging.ERROR)

from app.loaders.pdf_loader import load_pdf
from app.ingest import ingest_docs
from app.retriever import get_retriever
from app.graph import build_graph


docs = []
docs.extend(load_pdf("data/sample_pdfs/doc1.pdf"))
docs.extend(load_pdf("data/sample_pdfs/doc2.pdf"))

vector_db = ingest_docs(docs)
retriever = get_retriever(vector_db)
chatbot = build_graph(retriever)

state = {
    "question": "What is Retrieval-Augmented Generation in LLMs?",
    "context": [],
    "answer": "",
    "history": []
}

result = chatbot.invoke(state)
print("Answer:", result["answer"])
