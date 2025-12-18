from langchain_community.chat_models import ChatOllama

llm = ChatOllama(
    model="mistral", 
    temperature=0.1
)