from transformers import pipeline

generator = pipeline(
    "text2text-generation",
    model="google/flan-t5-small"
)


def rewrite_node(state):
    prompt = f"""
Rewrite the following question into a clear,
complete question that can be answered from documents.

Question: {state['question']}
"""
    result = generator(prompt, max_new_tokens=64)
    return {"question": result[0]["generated_text"].strip()}


def retrieve_node(state, retriever):
    docs = retriever.invoke(state["question"])

    docs = sorted(docs, key=lambda d: len(d.page_content), reverse=True)[:3]

    context = [d.page_content.replace("\n", " ").strip() for d in docs]

    return {"context": context}


def generate_node(state):
    prompt = f"""
You are a technical assistant.
Use ONLY the given context to answer.

Context:
{" ".join(state["context"])}

Question:
{state["question"]}
"""
    result = generator(prompt, max_new_tokens=180)
    return {"answer": result[0]["generated_text"].strip()}


def memory_node(state):
    history = state.get("history", [])
    history.append({
        "question": state["question"],
        "answer": state["answer"]
    })
    return {"history": history}
