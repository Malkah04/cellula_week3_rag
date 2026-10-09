from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from load_data import vector_store
from langchain_classic.memory import ConversationBufferMemory
from langchain_classic.chains import ConversationChain

import os
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("API_KEY")

llm = ChatOpenAI(
    api_key=api_key,
    base_url="https://openrouter.ai/api/v1",
    model="openrouter/free",
    temperature=0
)

memory = ConversationBufferMemory(
    memory_key="history"
)

prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You answer questions about Malak's CV.

Use only the information in the provided context.
If the answer is not available in the context, say that
you could not find the information in the CV.
Answer in clear English, using no more than 100 words.

Conversation history:
{history}

{input}

Answer:"""
)

chat = ConversationChain(
    llm=llm,
    memory=memory,
    prompt=prompt,
    verbose=False
)

query = "What defenses did Malak use against data poisoning?"

docs = vector_store.similarity_search_with_score(query, k=3)

context = "\n\n".join(
    document.page_content
    for document, score in docs
)

chat_input = f"Context:{context} Question:{query}"

res = chat.predict(input=chat_input)

print("Retrieved context:\n", context)
print("\nAnswer:\n", res)

print("\nSources:")
for document, score in docs:
    print("File:", document.metadata.get("source_file", "Unknown"))
    print("Section:", document.metadata.get("section", "Unknown"))
    print("Item:", document.metadata.get("item", "General section"))
    print("-" * 40)